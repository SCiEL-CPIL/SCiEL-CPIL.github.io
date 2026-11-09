"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from heapq import heappush, heappop
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime factor masks for each candidate
    masks = [0] * N
    scores = [0.0] * N
    for a in range(1, N):
        n = a
        mask = 0
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        masks[a] = mask
        scores[a] = 1.0 / (N - a)
        
    # Greedy construction helper
    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        used_mask = 0
        for a in ordering:
            if masks[a] & used_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                used_mask |= masks[a]
        return chosen, chosen_set, used_mask

    # Multiple initializations
    best_chosen = []
    best_chosen_set = set()
    best_used_mask = 0
    best_score = -1.0

    # 1. Descending score (a descending)
    ordering1 = list(range(N-1, 0, -1))
    ch, cs, cm = greedy_construct(ordering1)
    s = sum(scores[a] for a in ch)
    if s > best_score:
        best_score = s
        best_chosen = ch
        best_chosen_set = cs
        best_used_mask = cm

    # 2. Random with bias towards high scores
    for _ in range(10):
        if time.time() > deadline:
            break
        ordering = list(range(1, N))
        random.shuffle(ordering)
        # Bias: sort by random + score
        ordering.sort(key=lambda x: random.random() * 0.5 + scores[x], reverse=True)
        ch, cs, cm = greedy_construct(ordering)
        s = sum(scores[a] for a in ch)
        if s > best_score:
            best_score = s
            best_chosen = ch
            best_chosen_set = cs
            best_used_mask = cm

    # 3. Score / number of prime factors
    def num_pf(a):
        return bin(masks[a]).count('1')
    ordering3 = sorted(range(1, N), key=lambda x: scores[x] / max(1, num_pf(x)), reverse=True)
    ch, cs, cm = greedy_construct(ordering3)
    s = sum(scores[a] for a in ch)
    if s > best_score:
        best_score = s
        best_chosen = ch
        best_chosen_set = cs
        best_used_mask = cm

    chosen = best_chosen
    chosen_set = best_chosen_set
    used_mask = best_used_mask
    current_score = best_score

    # Local Search
    def local_search():
        nonlocal chosen, chosen_set, used_mask, current_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1-Swap
            non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
            # Sort non_chosen by score descending for faster pruning in 2-swap later
            non_chosen.sort(key=lambda x: scores[x], reverse=True)
            
            for i, a in enumerate(chosen):
                if time.time() > deadline:
                    break
                mask_without_a = used_mask ^ masks[a]
                best_b = -1
                best_gain = 0.0
                
                # Iterate non_chosen (already sorted by score desc)
                for b in non_chosen:
                    if masks[b] & mask_without_a == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    # Pruning: if scores[b] <= scores[a], and sorted, we can break?
                    # No, because we need gain > 0, so scores[b] > scores[a].
                    # Since sorted desc, once scores[b] <= scores[a], all subsequent are <=.
                    if scores[b] <= scores[a]:
                        break
                
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    used_mask = mask_without_a | masks[best_b]
                    current_score += best_gain
                    improved = True
                    # Update non_chosen list
                    # Remove best_b, add a
                    # Since we restart the while loop, we will rebuild non_chosen
                    break # Restart 1-swap loop
            
            if improved:
                continue # Restart while loop
            
            # 2-Swap
            # Rebuild non_chosen if needed, but we just built it above if 1-swap didn't trigger restart
            # However, if 1-swap didn't improve, non_chosen is still valid from the top of the loop
            # But we should ensure it's sorted and up to date.
            non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
            non_chosen.sort(key=lambda x: scores[x], reverse=True)
            
            n_chosen = len(chosen)
            found_2swap = False
            
            # Limit search to top K candidates to balance speed/quality
            K = 200
            top_non_chosen = non_chosen[:K]
            
            for i in range(n_chosen):
                if time.time() > deadline:
                    break
                a = chosen[i]
                for j in range(i+1, n_chosen):
                    c = chosen[j]
                    mask_without_ac = used_mask ^ masks[a] ^ masks[c]
                    target_score = scores[a] + scores[c]
                    
                    best_pair = None
                    best_gain = 0.0
                    
                    # Search in top_non_chosen
                    for b_idx, b in enumerate(top_non_chosen):
                        if masks[b] & mask_without_ac != 0:
                            continue
                        # Pruning: if scores[b] + max_possible_d <= target_score, break?
                        # max_possible_d is scores[top_non_chosen[0]] if b is not 0, or next.
                        # Simpler: just iterate.
                        for d_idx in range(b_idx + 1, len(top_non_chosen)):
                            d = top_non_chosen[d_idx]
                            if masks[d] & mask_without_ac != 0:
                                continue
                            if masks[b] & masks[d] != 0:
                                continue
                            gain = scores[b] + scores[d] - target_score
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (b, d)
                            # Pruning: if scores[b] + scores[d] <= target_score and sorted, 
                            # we can break inner loop? 
                            # Since d_idx increases, scores[d] decreases.
                            # If scores[b] + scores[d] <= target_score, then for larger d_idx, it will be even smaller.
                            if scores[b] + scores[d] <= target_score:
                                break
                    
                    if best_pair is not None and best_gain > 1e-12:
                        b, d = best_pair
                        chosen[i] = b
                        chosen[j] = d
                        chosen_set.remove(a)
                        chosen_set.remove(c)
                        chosen_set.add(b)
                        chosen_set.add(d)
                        used_mask = mask_without_ac | masks[b] | masks[d]
                        current_score += best_gain
                        improved = True
                        found_2swap = True
                        break # Restart while loop
                if found_2swap:
                    break

    local_search()

    # Branch and Bound for global optimum
    # We model this as Maximum Weight Independent Set on a conflict graph
    # Nodes: integers 1..511, Weight: 1/(512-a)
    # Edges: connect non-coprime pairs
    
    # Precompute conflict adjacency as bitsets for fast checking
    # Since we have ~97 primes, we can use the mask approach
    # Two numbers are coprime iff their prime factor masks are disjoint
    
    # Sort candidates by score descending for B&B
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: scores[x], reverse=True)
    
    # Precompute for each candidate, which higher-scored candidates conflict with it
    # conflict_mask[i] = bitmask of candidates j < i that conflict with candidate i
    # This helps in B&B to quickly check conflicts
    
    # For B&B, we use a recursive approach with memoization-like pruning
    # State: (next_candidate_index, used_prime_mask)
    # But used_prime_mask can be up to 2^97 bits, too large for memoization
    # Instead, we use branch and bound with tight upper bounds
    
    # Upper bound: sum of scores of all remaining candidates that are coprime to used_mask
    # This is a valid upper bound but may be loose
    
    # Better upper bound: LP relaxation or fractional knapsack
    # For each prime p, we can pick at most one number divisible by p
    # This gives a tighter bound via LP relaxation
    
    def branch_and_bound():
        nonlocal best_score, best_chosen, best_chosen_set, best_used_mask
        
        # Candidates sorted by score descending
        # We'll use a stack-based DFS with pruning
        
        # For efficiency, precompute for each position i, the list of candidates j > i 
        # that are coprime with candidate i
        # But this is O(n^2) space, which is ~511^2/2 ~ 130K, manageable
        
        # Actually, let's use a simpler approach:
        # At each node, we have:
        # - current_score: sum of scores of chosen candidates
        # - chosen_list: list of chosen candidates
        # - next_idx: next candidate to consider (index in sorted candidates)
        # - used_mask: prime factor mask of all chosen candidates
        
        # Upper bound: current_score + sum of scores of all candidates from next_idx onward
        # that are coprime to used_mask
        # This is tight enough for our problem size
        
        best_bb_score = best_score
        best_bb_chosen = list(best_chosen)
        
        # Use iterative deepening or just DFS with pruning
        # Since n=511, we need to be very careful about branching factor
        
        # Key insight: most candidates conflict with many others
        # The average number of coprime candidates should be small
        
        # Let's precompute for each candidate a, the count of candidates b > a 
        # that are coprime with a
        # This helps estimate the branching factor
        
        # For B&B, we'll use a priority queue (best-first) to explore most promising nodes first
        # Node: (negative_upper_bound, next_idx, used_mask, chosen_list, current_score)
        
        # But storing chosen_list in each node is expensive
        # Instead, we can reconstruct the solution from the path
        
        # Given time constraints, let's use a simpler approach:
        # 1. Get a good initial solution via greedy (already done)
        # 2. Use local search to improve (already done)
        # 3. Try a limited B&B with depth limit
        
        # For B&B, we'll limit the search depth and use aggressive pruning
        
        # Precompute coprime counts for pruning
        # coprime_count[i] = number of candidates j > i that are coprime with candidate i
        coprime_count = [0] * N
        for i in range(N):
            a = candidates[i]
            ma = masks[a]
            count = 0
            for j in range(i + 1, N):
                b = candidates[j]
                if masks[b] & ma == 0:
                    count += 1
            coprime_count[i] = count
        
        # B&B with recursion limit
        import sys
        sys.setrecursionlimit(10000)
        
        def bb_search(idx, used_mask, current_score, chosen_list):
            nonlocal best_bb_score, best_bb_chosen
            
            # Check time
            if time.time() > deadline - 5:  # 5s safety margin
                return
            
            # Upper bound: current_score + sum of scores of coprime candidates from idx onward
            ub = current_score
            for j in range(idx, N):
                a = candidates[j]
                if masks[a] & used_mask == 0:
                    ub += scores[a]
            
            if ub <= best_bb_score + 1e-12:
                return
            
            if idx >= N:
                if current_score > best_bb_score:
                    best_bb_score = current_score
                    best_bb_chosen = list(chosen_list)
                return
            
            a = candidates[idx]
            ma = masks[a]
            
            # Option 1: Include a
            if ma & used_mask == 0:
                chosen_list.append(a)
                bb_search(idx + 1, used_mask | ma, current_score + scores[a], chosen_list)
                chosen_list.pop()
            
            # Option 2: Exclude a
            bb_search(idx + 1, used_mask, current_score, chosen_list)
        
        # Start B&B
        bb_search(0, 0, 0.0, [])
        
        if best_bb_score > current_score:
            # Update best solution
            # Reconstruct the solution
            # We need to track the actual chosen set
            # For now, just update the score if B&B found better
            pass
    
    # Given the complexity, let's skip full B&B and instead do a more targeted improvement
    # Try adding elements one by one in a smart order
    
    def try_add_elements():
        nonlocal chosen, chosen_set, used_mask, current_score
        
        # Sort non-chosen by score descending
        non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
        non_chosen.sort(key=lambda x: scores[x], reverse=True)
        
        # Try to add elements in order of score
        for b in non_chosen:
            if time.time() > deadline - 5:
                break
            if masks[b] & used_mask == 0:
                # Can add b
                chosen.append(b)
                chosen_set.add(b)
                used_mask |= masks[b]
                current_score += scores[b]
    
    # Try adding elements
    try_add_elements()
    
    # Final local search polish
    def final_polish():
        nonlocal chosen, chosen_set, used_mask, current_score
        
        # Try 1-add moves
        non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
        non_chosen.sort(key=lambda x: scores[x], reverse=True)
        
        for b in non_chosen:
            if time.time() > deadline - 2:
                break
            if masks[b] & used_mask == 0:
                chosen.append(b)
                chosen_set.add(b)
                used_mask |= masks[b]
                current_score += scores[b]
        
        # Try 1-swap moves
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            non_chosen = [a for a in range(N-1, 0, -1) if a not in chosen_set]
            non_chosen.sort(key=lambda x: scores[x], reverse=True)
            
            for i, a in enumerate(chosen):
                if time.time() > deadline - 1:
                    break
                mask_without_a = used_mask ^ masks[a]
                best_b = -1
                best_gain = 0.0
                
                for b in non_chosen:
                    if masks[b] & mask_without_a == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    if scores[b] <= scores[a]:
                        break
                
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    used_mask = mask_without_a | masks[best_b]
                    current_score += best_gain
                    improved = True
                    break
    
    final_polish()

    return chosen
# EVOLVE-BLOCK-END
