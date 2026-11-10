"""QUBO + simulated annealing for 3-AP-free subset of [1,256] maximizing reciprocal sum."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    n = 256
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Precompute all 3-APs: list of (a, b, c) where a + c = 2*b, a < b < c
    # For QUBO, we need to know for each triple (i, j, k) with i+j+k pattern
    # Actually, for the QUBO formulation, we penalize any 3-AP.
    # A 3-AP is (a, b, c) with a + c = 2*b.
    # In binary variables x_i (1 if i is in set), we want to penalize x_a * x_b * x_c.
    # But QUBO is quadratic. We can use the fact that for a 3-AP (a,b,c),
    # the product x_a * x_c * x_b can be handled by noting that if x_a=x_c=1,
    # then x_b should be 0. We can add a penalty term P * x_a * x_c * x_b.
    # To make this quadratic, we can use a trick: for each 3-AP (a,b,c),
    # add penalty P * max(0, x_a + x_c - 1) * x_b, but that's not standard QUBO.
    
    # Alternative: use the constraint that for each 3-AP (a,b,c), not all three can be 1.
    # We can add penalty P * (x_a * x_b + x_b * x_c + x_a * x_c - 2*x_a*x_b*x_c)
    # But that's cubic. 
    
    # Better approach for SA: just check validity directly.
    # For each state (bitmask), compute the score and check if valid.
    # If invalid, apply a penalty.
    
    # Precompute for fast 3-AP checking:
    # For each pair (i, j) with i < j, compute the third element k = 2*j - i.
    # If k > j and k <= n, then (i, j, k) is a 3-AP.
    # Also (j, i, k) is not a 3-AP in the standard sense since we need i < j < k.
    
    # For checking if a set is 3-AP-free, we can iterate over all pairs and check.
    # But for SA, we need fast evaluation. Let's precompute a conflict matrix.
    
    # conflict[i][j] = list of k such that (i, j, k) is a 3-AP (in any order)
    # Actually, let's precompute for each pair (i, j), the set of k values
    # such that {i, j, k} forms a 3-AP.
    
    # For a set S, it's 3-AP-free iff for all i, j in S with i != j,
    # the third element k (if it exists and is in [1,n]) is not in S.
    
    # Precompute: for each i, j (i != j), what is k such that {i,j,k} is a 3-AP?
    # There are at most 3 possible k values for each pair (i,j):
    #   k1 = 2*i - j (i is middle)
    #   k2 = 2*j - i (j is middle)  
    #   k3 = (i + j) / 2 if (i+j) is even (k3 is middle)
    
    third_elems = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            candidates = set()
            # i is middle: k = 2*i - j
            k = 2 * i - j
            if 1 <= k <= n and k != i and k != j:
                candidates.add(k)
            # j is middle: k = 2*j - i
            k = 2 * j - i
            if 1 <= k <= n and k != i and k != j:
                candidates.add(k)
            # middle is (i+j)/2
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= n and k != i and k != j:
                    candidates.add(k)
            if candidates:
                third_elems[(i, j)] = candidates
    
    def is_3ap_free_mask(mask):
        """Check if bitmask represents a 3-AP-free set."""
        # Get all set bits
        elements = [i for i in range(1, n + 1) if (mask >> i) & 1]
        elem_set = set(elements)
        for idx, i in enumerate(elements):
            for j in elements[idx + 1:]:
                if (i, j) in third_elems:
                    for k in third_elems[(i, j)]:
                        if k in elem_set:
                            return False
        return True
    
    def reciprocal_sum_mask(mask):
        """Compute reciprocal sum of bitmask."""
        total = 0.0
        for i in range(1, n + 1):
            if (mask >> i) & 1:
                total += recip[i]
        return total

    # Greedy initialization
    def greedy_init():
        mask = 0
        selected = set()
        for x in range(1, n + 1):
            # Check if adding x creates a 3-AP
            valid = True
            for a in selected:
                # x is middle: a + c = 2*x => c = 2*x - a
                c = 2 * x - a
                if c in selected and c != x:
                    valid = False
                    break
                # x is endpoint, a is middle: x + c = 2*a => c = 2*a - x
                c = 2 * a - x
                if c in selected and c != x and c != a:
                    valid = False
                    break
                # a is endpoint, x is middle already checked
                # a is middle, x is other endpoint: a + x = 2*c => c = (a+x)/2
                if (a + x) % 2 == 0:
                    c = (a + x) // 2
                    if c in selected and c != x and c != a:
                        valid = False
                        break
            if valid:
                mask |= (1 << x)
                selected.add(x)
        return mask

    best_mask = greedy_init()
    best_score = reciprocal_sum_mask(best_mask)
    
    # Verify
    if not is_3ap_free_mask(best_mask):
        # Fallback: just use {1}
        best_mask = 1 << 1
        best_score = recip[1]

    # Simulated annealing
    def simulated_annealing(mask, score, deadline):
        """Run SA to improve the solution."""
        current_mask = mask
        current_score = score
        best_sa_mask = mask
        best_sa_score = score
        
        # Initial temperature
        T = 0.01
        cooling = 0.9995
        min_T = 1e-6
        
        while (time.time() - start_time) < deadline and T > min_T:
            # Random move: flip a random bit
            x = random.randint(1, n)
            new_mask = current_mask ^ (1 << x)
            
            # Check if new state is valid
            if not is_3ap_free_mask(new_mask):
                # Invalid: reject (or accept with very low probability)
                # For now, just reject invalid moves
                T *= cooling
                continue
            
            new_score = reciprocal_sum_mask(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_sa_score:
                    best_sa_mask = current_mask
                    best_sa_score = current_score
            
            T *= cooling
            
            # Occasionally try multiple flips
            if random.random() < 0.1:
                # Try flipping 2 bits
                x1 = random.randint(1, n)
                x2 = random.randint(1, n)
                if x1 != x2:
                    new_mask = current_mask ^ (1 << x1) ^ (1 << x2)
                    if is_3ap_free_mask(new_mask):
                        new_score = reciprocal_sum_mask(new_mask)
                        delta = new_score - current_score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            current_mask = new_mask
                            current_score = new_score
                            if current_score > best_sa_score:
                                best_sa_mask = current_mask
                                best_sa_score = current_score
        
        return best_sa_mask, best_sa_score

    # Run SA with multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if (time.time() - start_time) > DEADLINE - 10.0:
            break
        
        if restart == 0:
            start_mask = best_mask
            start_score = best_score
        else:
            # Perturb the best solution
            start_mask = best_mask
            # Remove 3-5 random elements
            elements = [i for i in range(1, n + 1) if (start_mask >> i) & 1]
            if len(elements) > 5:
                to_remove = random.sample(elements, min(3, len(elements) // 2))
                for x in to_remove:
                    start_mask &= ~(1 << x)
            start_score = reciprocal_sum_mask(start_mask)
        
        sa_mask, sa_score = simulated_annealing(start_mask, start_score, DEADLINE - 5.0)
        if sa_score > best_score:
            best_mask = sa_mask
            best_score = sa_score

    # Final local search polish
    def local_search_polish(mask, score, deadline):
        """Try single-bit flips to improve."""
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for x in range(1, n + 1):
                # Try adding x if not present
                if not ((mask >> x) & 1):
                    new_mask = mask | (1 << x)
                    if is_3ap_free_mask(new_mask):
                        new_score = reciprocal_sum_mask(new_mask)
                        if new_score > score + 1e-15:
                            mask = new_mask
                            score = new_score
                            improved = True
                # Try removing x if present
                else:
                    new_mask = mask & ~(1 << x)
                    new_score = reciprocal_sum_mask(new_mask)
                    # Only remove if it allows adding something better
                    # For now, just check if removal helps (it won't directly, but might enable additions)
                    # Skip removals for simplicity in polish
        return mask, score

    best_mask, best_score = local_search_polish(best_mask, best_score, DEADLINE - 2.0)

    # Final verification
    if not is_3ap_free_mask(best_mask):
        # Emergency fallback
        best_mask = 1 << 1
        best_score = recip[1]

    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    assert len(result) > 0, "Solution is empty!"
    assert is_3ap_free_mask(best_mask), "Final solution has a 3-AP!"
    
    return result
# EVOLVE-BLOCK-END
