"""Enhanced search for the 'no element divides sum of three others' problem."""
import time
import random
from itertools import combinations
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    N = 192
    
    # Precompute: for each a, which triples (b,c,d) with b<c<d, a not in {b,c,d}
    # satisfy a | (b+c+d). We'll use this to check conflicts.
    # Actually, let's precompute for each pair of "summands" what sums they create,
    # then check divisibility.
    
    # A more efficient approach: precompute for each a, the set of sums s such that
    # a divides s. Then when checking if we can add a to a set S, we need to verify
    # that for no triple (b,c,d) in S does (b+c+d) % a == 0.
    
    # Even better: precompute all triples and their sums.
    # There are C(192,3) ≈ 1.15M triples. That's a lot but manageable.
    
    # Alternative: For each candidate a, and each pair (b,c) in current set,
    # check if there exists d in current set, d != b,c, such that (b+c+d) % a == 0.
    # This is O(|S|^2) per candidate, which is O(n^2) total for greedy.
    
    def is_valid_addition(a, current_set):
        """Check if adding a to current_set maintains the property."""
        # a must not divide sum of any 3 distinct other elements in current_set
        # Also, for each element x in current_set, x must not divide sum of any
        # 3 distinct elements from (current_set + [a]) excluding x.
        # This means:
        # 1. a doesn't divide sum of any 3 from current_set
        # 2. For each x in current_set, x doesn't divide sum of any 3 from (current_set - {x}) ∪ {a}
        #    This breaks into:
        #    a) x doesn't divide sum of 3 from current_set - {x} (already true by invariant)
        #    b) x doesn't divide (a + sum of 2 from current_set - {x})
        #    c) x doesn't divide (sum of 1 from current_set - {x} + ... wait, we need 3 distinct others)
        #    When we add a, the new triples involving a and x as the tested element:
        #    For tested element x, new triples are those that include a and 2 others from current_set - {x}
        #    So we need: for all x in current_set, for all pairs (u,v) in current_set - {x},
        #    (x + a + u + v) ... no wait.
        #    
        #    Let me re-think. The condition is: for no selected integer a, does a divide
        #    the sum of any 3 DISTINCT OTHER selected integers.
        #    
        #    When we add a to S to get S' = S ∪ {a}:
        #    - For a itself: a must not divide sum of any 3 from S.
        #    - For each x in S: x must not divide sum of any 3 from S' \ {x} = (S \ {x}) ∪ {a}.
        #      The triples from (S \ {x}) ∪ {a} are:
        #      - 3 from S \ {x}: already checked
        #      - 2 from S \ {x} and a: need to check x ∤ (a + u + v) for all u,v in S\{x}, u≠v
        #      - 1 from S \ {x} and ... wait, we need 3 distinct. So:
        #        - {a, u, v} where u,v in S\{x}, u≠v
        #        That's it for new triples involving a.
        #      So we need: for all x in S, for all {u,v} ⊂ S\{x}, x ∤ (a + u + v)
        
        # Check 1: a doesn't divide sum of any 3 from S
        if len(current_set) >= 3:
            for triple in combinations(current_set, 3):
                if sum(triple) % a == 0:
                    return False
        
        # Check 2: for each x in S, x doesn't divide (a + u + v) for any u,v in S\{x}
        S_list = list(current_set)
        for i, x in enumerate(S_list):
            # Pairs from S_list excluding x
            others = S_list[:i] + S_list[i+1:]
            for j in range(len(others)):
                for k in range(j+1, len(others)):
                    if (a + others[j] + others[k]) % x == 0:
                        return False
        return True
    
    # Greedy large-first baseline
    def greedy_large_first():
        selected = []
        for x in range(N, 0, -1):
            if is_valid_addition(x, selected):
                selected.append(x)
        return selected
    
    # Try to improve with local search: swap moves
    def local_search(initial_set, max_iters=5000):
        S = list(initial_set)
        S_set = set(S)
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try removing one element and adding two
            for i in range(len(S)):
                if time.time() - start_time > DEADLINE:
                    break
                removed = S[i]
                S_without = S[:i] + S[i+1:]
                # Try adding two new elements
                candidates = [x for x in range(N, 0, -1) if x not in S_set and x != removed]
                # Sort by value descending
                candidates.sort(reverse=True)
                
                found_pair = False
                for a in candidates:
                    if found_pair or time.time() - start_time > DEADLINE:
                        break
                    if not is_valid_addition(a, S_without):
                        continue
                    S_with_a = S_without + [a]
                    for b in candidates:
                        if b <= a:
                            continue
                        if not is_valid_addition(b, S_with_a):
                            continue
                        # Check if we can add b
                        new_S = S_with_a + [b]
                        if len(new_S) > best_size:
                            S = new_S
                            S_set = set(S)
                            best_size = len(S)
                            improved = True
                            found_pair = True
                            break
                        elif len(new_S) == best_size:
                            # Accept with some probability for diversity
                            if random.random() < 0.1:
                                S = new_S
                                S_set = set(S)
                                improved = True
                                found_pair = True
                                break
                if found_pair:
                    break
            
            # Also try single swaps: remove one, add one
            if not improved:
                for i in range(len(S)):
                    if time.time() - start_time > DEADLINE:
                        break
                    removed = S[i]
                    S_without = S[:i] + S[i+1:]
                    candidates = [x for x in range(N, 0, -1) if x not in S_set and x != removed]
                    candidates.sort(reverse=True)
                    
                    for a in candidates:
                        if time.time() - start_time > DEADLINE:
                            break
                        if is_valid_addition(a, S_without):
                            new_S = S_without + [a]
                            if len(new_S) == best_size:
                                # Same size, accept with probability
                                if random.random() < 0.05:
                                    S = new_S
                                    S_set = set(S)
                                    improved = True
                                    break
                            elif len(new_S) > best_size:
                                S = new_S
                                S_set = set(S)
                                best_size = len(S)
                                improved = True
                                break
                    if improved:
                        break
        
        return S
    
    # Get baseline
    baseline = greedy_large_first()
    best_set = list(baseline)
    best_size = len(best_set)
    
    # Run local search from baseline
    if time.time() - start_time < DEADLINE:
        improved = local_search(baseline, max_iters=2000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Try different random seeds for greedy to get diverse starting points
    random.seed(42)
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Randomized greedy
        selected = []
        candidates = list(range(N, 0, -1))
        random.shuffle(candidates)
        # But bias towards larger numbers
        candidates.sort(key=lambda x: -x + random.random() * 20)
        
        for x in candidates:
            if is_valid_addition(x, selected):
                selected.append(x)
        
        if len(selected) > best_size:
            best_set = selected
            best_size = len(selected)
        
        # Local search from this starting point
        if time.time() - start_time < DEADLINE - 5:
            improved = local_search(selected, max_iters=1000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    # Final check and return
    # Verify the result is valid
    def verify(S):
        S_set = set(S)
        for a in S:
            others = [x for x in S if x != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    if not verify(best_set):
        # Fallback to baseline if something went wrong
        best_set = baseline
    
    return best_set
