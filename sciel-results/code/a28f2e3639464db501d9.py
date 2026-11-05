"""Combined program: exact B&B solver with local search fallback."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, p):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Precompute sorted candidates by score descending
    candidates_by_score = sorted(range(511, 0, -1), key=lambda x: scores[x], reverse=True)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def valid(s):
        used = set()
        for a in s:
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True
    
    def find_best_add(used_primes, current_set, exclude=None):
        """Find the best number to add that is coprime to used_primes and not in current_set."""
        if exclude is None:
            exclude = set()
        for a in candidates_by_score:
            if a in current_set or a in exclude:
                continue
            if not (prime_factors[a] & used_primes):
                return a
        return None

    def hill_climb(initial_set, max_iters=5000):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue

            # 2. Try 1-for-1 swap: remove one, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                # Find best pair to add
                best_pair = None
                best_pair_score = 0
                
                # Get list of valid candidates
                valid_cands = []
                for a in candidates_by_score:
                    if a in rem_set:
                        continue
                    if prime_factors[a] & rem_used:
                        continue
                    valid_cands.append(a)
                    if len(valid_cands) > 50: # Limit search space for pairs
                        break
                
                for idx_a in range(len(valid_cands)):
                    a = valid_cands[idx_a]
                    for idx_b in range(idx_a + 1, len(valid_cands)):
                        b = valid_cands[idx_b]
                        if prime_factors[a] & prime_factors[b]:
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_set = rem_set | set(best_pair)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if improved:
                continue

            # 4. Try 2-for-1 swap: remove two, add one (MERGE)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = set()
                    for a in remaining:
                        rem_used |= prime_factors[a]
                    
                    best_add = find_best_add(rem_used, rem_set)
                    if best_add is not None:
                        new_score = rem_score + scores[best_add]
                        if new_score > current_score + 1e-15:
                            current = remaining + [best_add]
                            current_set = rem_set | {best_add}
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return current, current_score
    
    def branch_and_bound(initial_set, time_budget=120.0):
        """
        Exact B&B solver using greedy coloring upper bounds.
        The conflict graph: two numbers conflict if they share a prime factor.
        We want maximum weight independent set.
        """
        bb_start = time.time()
        best_score_bb = get_score(initial_set)
        best_set_bb = list(initial_set)
        
        # Build conflict graph: for each number, list of numbers it conflicts with
        conflict = [set() for _ in range(512)]
        for a in range(1, 512):
            for b in range(a+1, 512):
                if prime_factors[a] & prime_factors[b]:
                    conflict[a].add(b)
                    conflict[b].add(a)
        
        # Precompute neighbors for each node
        neighbors = [list(conflict[a]) for a in range(512)]
        
        # Greedy coloring for upper bound
        def greedy_color_bound(candidate_list, used_primes):
            """
            Compute upper bound using greedy coloring.
            Colors represent groups of mutually non-conflicting nodes.
            The sum of max scores in each color group is an upper bound.
            """
            if not candidate_list:
                return 0.0
            
            # Filter to only valid candidates (coprime to used_primes)
            valid = [a for a in candidate_list if not (prime_factors[a] & used_primes)]
            if not valid:
                return 0.0
            
            # Sort by score descending
            valid_sorted = sorted(valid, key=lambda x: scores[x], reverse=True)
            
            # Greedy coloring: assign colors such that no two conflicting nodes share a color
            color_groups = {}
            node_color = {}
            
            for a in valid_sorted:
                # Find the first color that doesn't conflict with any node of that color
                assigned = False
                for color in color_groups:
                    # Check if a conflicts with any node in this color group
                    conflicts = False
                    for b in color_groups[color]:
                        if a in conflict[b]:
                            conflicts = True
                            break
                    if not conflicts:
                        color_groups[color].append(a)
                        node_color[a] = color
                        assigned = True
                        break
                
                if not assigned:
                    new_color = len(color_groups)
                    color_groups[new_color] = [a]
                    node_color[a] = new_color
            
            # Upper bound: sum of max scores in each color group
            bound = 0.0
            for color, nodes in color_groups.items():
                bound += max(scores[a] for a in nodes)
            
            return bound
        
        # B&B search
        def bb_search(current_set, current_score, candidate_list, depth=0):
            nonlocal best_score_bb, best_set_bb
            
            if time.time() > bb_start + time_budget:
                return
            
            # Update best if we have a better solution
            if current_score > best_score_bb:
                best_score_bb = current_score
                best_set_bb = list(current_set)
            
            if not candidate_list:
                return
            
            # Compute upper bound
            used_primes = set()
            for a in current_set:
                used_primes |= prime_factors[a]
            
            ub = current_score + greedy_color_bound(candidate_list, used_primes)
            
            # Prune if upper bound is not better than best
            if ub <= best_score_bb + 1e-15:
                return
            
            # Select next node to branch on (highest score)
            valid_candidates = [a for a in candidate_list if not (prime_factors[a] & used_primes)]
            if not valid_candidates:
                return
            
            # Sort by score descending
            valid_candidates.sort(key=lambda x: scores[x], reverse=True)
            next_node = valid_candidates[0]
            
            # Branch 1: include next_node
            new_current_set = set(current_set)
            new_current_set.add(next_node)
            new_used_primes = used_primes | prime_factors[next_node]
            
            # Filter candidates that don't conflict with next_node
            new_candidates = [a for a in valid_candidates[1:] if not (prime_factors[a] & prime_factors[next_node])]
            
            bb_search(new_current_set, current_score + scores[next_node], new_candidates, depth+1)
            
            # Branch 2: exclude next_node
            remaining_candidates = valid_candidates[1:]
            bb_search(current_set, current_score, remaining_candidates, depth+1)
        
        # Start B&B
        initial_candidates = list(range(1, 512))
        bb_search(set(initial_set), get_score(initial_set), initial_candidates)
        
        return best_set_bb, best_score_bb
    
    # Initial strategy: Start with all primes
    initial_set = [p for p in range(2, 512) if is_prime[p]]
    
    # First, run local search to get a good starting point
    best_set, best_score = hill_climb(initial_set, max_iters=2000)
    
    # Run B&B with the local search result as initial solution
    bb_set, bb_score = branch_and_bound(best_set, time_budget=100.0)
    
    if bb_score > best_score + 1e-15:
        best_set = bb_set
        best_score = bb_score
    
    # Final polish with local search if time permits
    if time.time() < deadline - 5:
        result, score = hill_climb(best_set, max_iters=5000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    return best_set
# EVOLVE-BLOCK-END
