"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Numbers are in [2, 256]
    N = 255
    # Index i corresponds to number i + 2
    
    # Precompute weights
    weights = [1.0 / (i + 2) for i in range(N)]
    
    # Precompute compatibility bitmasks
    # compat_mask[i] has bit j set if lcm(i+2, j+2) > 256
    # We want to find a Maximum Weight Clique in this graph.
    # A clique in the compatibility graph is a set of numbers where every pair has lcm > 256.
    
    compat_mask = [0] * N
    
    for i in range(N):
        a = i + 2
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 2
            g = gcd(a, b)
            # lcm(a, b) = a * b // g
            if a * b // g > 256:
                mask |= (1 << j)
        compat_mask[i] = mask

    # To speed up bound calculation, we can precompute suffix sums of weights
    # if we process in a specific order, but BK doesn't strictly rely on order for correctness,
    # just for efficiency. We can sort vertices by weight to help pruning.
    # However, re-indexing complicates bitmasks. 
    # Alternative: Use the natural index but pick pivots wisely.
    
    # Let's try sorting indices by weight descending and re-mapping.
    # This allows us to use simple "sum of remaining weights" as a bound more effectively
    # if we process in that order? No, BK explores subsets.
    
    # Actually, a simpler bound is just sum(weights[v] for v in candidates).
    # We can compute this quickly if we maintain a list or use precomputed sums?
    # With bitmasks, summing weights of set bits is O(N) per node in worst case.
    # N=255. 255 * (number of nodes). If number of nodes is < 10^5, it's 25M ops, which is fine in 85s.
    
    best_weight = 0.0
    best_clique = 0  # Bitmask of the best clique
    
    # Recursive function for Weighted Bron-Kerbosch
    # P: candidates, X: excluded, R: current clique (bitmask), current_w: weight of R
    def bk(P, X, R, current_w):
        nonlocal best_weight, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if current_w > best_weight:
                best_weight = current_w
                best_clique = R
            return
        
        # Bound: Sum of weights of all vertices in P
        # If current_w + sum(weights[v] for v in P) <= best_weight, prune
        # Optimization: If we sort vertices by weight, we could precompute prefix sums,
        # but P is a bitmask. 
        # Let's just iterate.
        bound = current_w
        temp_P = P
        while temp_P:
            # Get least significant set bit
            lsb = temp_P & -temp_P
            idx = lsb.bit_length() - 1
            bound += weights[idx]
            temp_P ^= lsb
            
        if bound <= best_weight:
            return
            
        # Pivot selection
        # Choose u from P | X with maximum degree in the subgraph induced by P | X?
        # Or just from P? Standard BK uses P | X.
        # To maximize pruning, we want to minimize the number of branches.
        # Branches are for v in P \ N(u).
        # So we want u to have a large N(u) intersection with P.
        
        P_or_X = P | X
        if P_or_X == 0:
            # Should be handled by P==0 check above, but safe
            if current_w > best_weight:
                best_weight = current_w
                best_clique = R
            return

        # Find pivot u
        pivot = -1
        max_count = -1
        temp_PX = P_or_X
        while temp_PX:
            lsb = temp_PX & -temp_PX
            u = lsb.bit_length() - 1
            # Count neighbors of u in P
            # N(u) in P is (compat_mask[u] & P)
            count = (compat_mask[u] & P).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            temp_PX ^= lsb
            
        # Branch on vertices in P \ N(pivot)
        # If pivot is -1 (should not happen if P_or_X != 0), branch on all P
        if pivot == -1:
            candidates_to_branch = P
        else:
            candidates_to_branch = P & ~(compat_mask[pivot])
            
        temp_cand = candidates_to_branch
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            
            # Include v in clique
            new_R = R | lsb
            new_w = current_w + weights[v]
            
            # New P: neighbors of v in current P
            new_P = P & compat_mask[v]
            # New X: vertices in X that are neighbors of v? 
            # Standard BK: X' = X. P' = P \ N(v). 
            # Wait, the standard recursive step is:
            # BK(R ∪ {v}, P ∩ N(v), X ∩ N(v))
            # This ensures we don't revisit cliques.
            new_X = X & compat_mask[v]
            
            bk(new_P, new_X, new_R, new_w)
            
            # Remove v from P, add to X
            P ^= lsb
            X |= lsb
            temp_cand ^= lsb
            
        # Note: The loop above modifies P and X. 
        # The standard algorithm iterates through vertices in P \ N(u).
        # For each v, we recurse, then remove v from P and add to X.
        # This is correct.

    # Initial call
    # P = all vertices, X = empty
    initial_P = (1 << N) - 1
    bk(initial_P, 0, 0, 0.0)
    
    # Extract result from best_clique bitmask
    result_numbers = []
    for i in range(N):
        if best_clique & (1 << i):
            result_numbers.append(i + 2)
            
    # Fallback if no clique found (shouldn't happen as single nodes are valid)
    if not result_numbers:
        # Find the single number with max weight (smallest number)
        # Actually, any single number is a valid clique of size 1.
        # The smallest number 2 has weight 0.5.
        # Let's just return [2] as a safe fallback, though BK should find better.
        # Wait, if BK found nothing, it means P was empty initially? No.
        # It means best_weight was 0. 
        # Let's ensure we at least return the best single node.
        max_w = 0
        max_idx = 0
        for i in range(N):
            if weights[i] > max_w:
                max_w = weights[i]
                max_idx = i
        result_numbers = [max_idx + 2]

    # Final verification
    for i in range(len(result_numbers)):
        for j in range(i+1, len(result_numbers)):
            a = result_numbers[i]
            b = result_numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                # Invalid, fallback to greedy
                # Greedy: pick highest weight compatible
                greedy_set = []
                greedy_mask = 0
                for i in range(N):
                    if (compat_mask[i] & greedy_mask) == 0:
                        greedy_set.append(i + 2)
                        greedy_mask |= (1 << i)
                return greedy_set
                
    return result_numbers

# EVOLVE-BLOCK-END
