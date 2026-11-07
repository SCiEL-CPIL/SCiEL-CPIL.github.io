"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. The condition "all subset products are distinct" is equivalent to the set of prime exponent vectors being "dissociated" (no non-trivial linear combination with coefficients -1, 0, 1 sums to 0).
    2. This implies that no minimal subset of the vectors is affinely dependent in a specific way (e.g., sum of some equals sum of others).
    3. We precompute all "bad" minimal subsets (hyperedges) of size up to 4 or 5.
    4. We use a branch-and-bound search that adds vectors one by one, checking if the new vector forms a bad subset with any existing subset of the current set.
    5. This check is local and fast, allowing for a deeper search than checking all 2^k sums.
    """
    
    candidates = list(range(2, 41))
    
    # 1. Precompute prime exponent vectors for each candidate
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        v = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)
        
    vectors = {x: get_vector(x) for x in candidates}
    indices = list(range(len(candidates)))
    cand_idx_map = {x: i for i, x in enumerate(candidates)}
    
    # 2. Identify all minimal "bad" sets (hyperedges)
    # A set is bad if there exist disjoint non-empty subsets P, N such that sum(P) = sum(N).
    # We look for small sets (size 3 and 4).
    bad_sets = set()
    
    # Check triples: v_a + v_b = v_c
    # Note: v_a, v_b, v_c are distinct candidates.
    # We store the set of indices.
    for i in indices:
        for j in indices:
            if i == j: continue
            for k in indices:
                if i == k or j == k: continue
                # Check if v_i + v_j == v_k
                # Since vectors are positive, if v_i + v_j == v_k, then v_k is "larger".
                # We just check equality.
                vi = vectors[candidates[i]]
                vj = vectors[candidates[j]]
                vk = vectors[candidates[k]]
                sum_ij = tuple(vi[t] + vj[t] for t in range(num_primes))
                if sum_ij == vk:
                    bad_sets.add(frozenset([i, j, k]))
                # Also check v_i + v_k == v_j, etc.
                # But due to symmetry of the check (sum of two equals third), 
                # checking all pairs (i,j) and comparing to k covers all cases where one is the sum of two others.
                # We don't need to check v_i == v_j + v_k separately if we iterate all permutations?
                # My loop checks v_i + v_j == v_k.
                # If v_k = v_i + v_j, it's found.
                # If v_i = v_j + v_k, it's found when i,j,k are permuted such that the sum is the third.
                # Specifically, if v_i = v_j + v_k, then v_j + v_k = v_i.
                # In the loop, when we pick j, k as the pair and i as the target, we check v_j + v_k == v_i.
                # So all 1+1=1 dependencies are covered.
                
    # Check quads: v_a + v_b = v_c + v_d
    # This covers dependencies where the split is 2 vs 2.
    # We also need to check if a quad contains a bad triple? 
    # If a quad is bad because of a triple, it's not minimal.
    # We only care about minimal bad sets for the hypergraph, but for the check "can we add x", 
    # we need to know if x + subset of S is bad.
    # If x + {a,b} is a bad triple, we catch it.
    # If x + {a,b,c} is a bad quad (2 vs 2), we need to catch it.
    
    for i in indices:
        for j in indices:
            if i == j: continue
            for k in indices:
                if i == k or j == k: continue
                for l in indices:
                    if i == l or j == l or k == l: continue
                    vi = vectors[candidates[i]]
                    vj = vectors[candidates[j]]
                    vk = vectors[candidates[k]]
                    vl = vectors[candidates[l]]
                    
                    # Check v_i + v_j == v_k + v_l
                    sum_ij = tuple(vi[t] + vj[t] for t in range(num_primes))
                    sum_kl = tuple(vk[t] + vl[t] for t in range(num_primes))
                    if sum_ij == sum_kl:
                        bad_sets.add(frozenset([i, j, k, l]))
                        
                    # Check v_i + v_k == v_j + v_l
                    sum_ik = tuple(vi[t] + vk[t] for t in range(num_primes))
                    sum_jl = tuple(vj[t] + vl[t] for t in range(num_primes))
                    if sum_ik == sum_jl:
                        bad_sets.add(frozenset([i, j, k, l]))
                        
                    # Check v_i + v_l == v_j + v_k
                    sum_il = tuple(vi[t] + vl[t] for t in range(num_primes))
                    sum_jk = tuple(vj[t] + vk[t] for t in range(num_primes))
                    if sum_il == sum_jk:
                        bad_sets.add(frozenset([i, j, k, l]))

    # Optimization: Group bad sets by their elements to speed up lookup?
    # Or just keep the set of frozensets.
    # To check if adding x to S creates a bad set:
    # We need to check if there exists a subset T of S such that T U {x} is in bad_sets.
    # Since bad sets are small (size 3 or 4), T must have size 2 or 3.
    # We can precompute for each index x, the list of bad sets containing x.
    bad_sets_containing = {i: [] for i in indices}
    for bs in bad_sets:
        for i in bs:
            bad_sets_containing[i].append(bs)

    # Sort candidates by a heuristic to find a good solution early.
    # Vectors with smaller "norm" or "support" might be easier to pack?
    # Or larger? 
    # Let's sort by the value itself, or by norm.
    def heuristic_score(x):
        v = vectors[x]
        return sum(v)
        
    sorted_indices = sorted(indices, key=lambda i: heuristic_score(candidates[i]))
    
    best_size = 0
    best_set = []
    
    def backtrack(start_idx, current_set_indices, current_set_vals):
        nonlocal best_size, best_set
        
        if time.time() > _DEADLINE:
            return
            
        if len(current_set_indices) > best_size:
            best_size = len(current_set_indices)
            best_set = list(current_set_vals)
            
        # Pruning: If we can't beat best_size, return
        if len(current_set_indices) + (len(sorted_indices) - start_idx) <= best_size:
            return
            
        for i in range(start_idx, len(sorted_indices)):
            if time.time() > _DEADLINE:
                return
                
            idx = sorted_indices[i]
            val = candidates[idx]
            
            # Check if adding idx to current_set_indices creates a bad set
            # We need to check if there is a bad set B such that B \ {idx} is a subset of current_set_indices
            # and idx is in B.
            valid = True
            for B in bad_sets_containing[idx]:
                # B is a frozenset of indices
                # Check if all other elements in B are in current_set_indices
                # Since B is small, this is fast.
                rest = B - frozenset([idx])
                if rest.issubset(frozenset(current_set_indices)):
                    valid = False
                    break
            
            if valid:
                backtrack(i + 1, current_set_indices + [idx], current_set_vals + [val])
                
    # Start search
    backtrack(0, [], [])
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
