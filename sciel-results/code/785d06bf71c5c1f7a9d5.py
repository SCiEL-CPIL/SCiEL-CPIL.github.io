import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate Type A vectors (Weight 4)
    pos_sets = list(itertools.combinations(range(D), 4))
    signs_list = list(itertools.product([1, -1], repeat=4))
    
    # Precompute properties for fast checking
    # Store: (vector_array, support_frozenset, signs_tuple)
    # Actually, we can just store the vector and compute support/signs on the fly or precompute.
    # Precomputing is faster.
    
    vectors_a = []
    supports_a = []
    signs_a = []
    
    for p in pos_sets:
        support = frozenset(p)
        for s in signs_list:
            v = [0] * D
            for pos, sig in zip(p, s):
                v[pos] = sig
            vectors_a.append(v)
            supports_a.append(support)
            signs_a.append(s)
            
    N_A = len(vectors_a)
    
    # Generate Type B vectors (Weight 1, norm 2)
    vectors_b = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            vectors_b.append(v)
            
    # All Type B are compatible with each other and all Type A.
    # So we always include all 22.
    
    best_size = 0
    best_selected = []
    
    # Randomized Greedy Search
    # We will try multiple random orderings
    
    # To speed up, we can use a bitset or just lists.
    # Given N=5280, list of lists is fine.
    
    while time.time() < deadline:
        # Create random order
        order = list(range(N_A))
        random.shuffle(order)
        
        selected_indices = []
        selected_vectors = []
        selected_supports = []
        selected_signs = []
        
        for idx in order:
            v = vectors_a[idx]
            sup = supports_a[idx]
            sig = signs_a[idx]
            
            is_compatible = True
            # Check against all selected
            for k in range(len(selected_vectors)):
                v_sel = selected_vectors[k]
                sup_sel = selected_supports[k]
                sig_sel = selected_signs[k]
                
                # Check if identical
                if sup == sup_sel:
                    if sig == sig_sel:
                        is_compatible = False
                        break
                    else:
                        # Same support, different signs.
                        # Dot product <= 2 always for weight 4 vectors with same support?
                        # Max dot is 4 (identical).
                        # If not identical, at least one sign differs.
                        # If 1 differs: 1+1+1-1 = 2.
                        # If 2 differ: 1+1-1-1 = 0.
                        # If 3 differ: -1-1-1+1 = -2.
                        # If 4 differ: -4.
                        # So always <= 2.
                        continue
                
                # Check intersection size 3
                inter = sup.intersection(sup_sel)
                if len(inter) == 3:
                    # Check if signs match on intersection
                    # We need to compare signs at positions in inter
                    # sig is tuple of 4 signs corresponding to p (sorted)
                    # sup is frozenset
                    # This is tricky to do fast with tuples.
                    
                    # Alternative: Store vector as array and do dot product?
                    # Dot product is fast in numpy.
                    # But we are in a Python loop.
                    
                    # Let's optimize the sign check.
                    # We can precompute for each vector a dictionary {position: sign}
                    # Or just use the vector array.
                    
                    # Let's use the vector array for the intersection check.
                    # v is list of 11 ints.
                    # v_sel is list of 11 ints.
                    # Sum of products on inter.
                    # This is 3 multiplications and additions.
                    dot_inter = 0
                    for pos in inter:
                        dot_inter += v[pos] * v_sel[pos]
                    
                    if dot_inter == 3: # Signs match
                        is_compatible = False
                        break
                    elif dot_inter == -3: # Signs opposite
                        continue
                    else: # 1 or -1
                        continue
                else:
                    # Intersection size 0, 1, 2, 4
                    # If 4, handled above.
                    # If 0, 1, 2, max dot is 2.
                    continue
            
            if is_compatible:
                selected_indices.append(idx)
                selected_vectors.append(v)
                selected_supports.append(sup)
                selected_signs.append(sig)
        
        if len(selected_indices) > best_size:
            best_size = len(selected_indices)
            best_selected = selected_indices
            
            # Optional: If we find a very good set, we can stop early?
            # Or just keep trying to beat it.
            
    # Construct final array
    result_a = np.array([vectors_a[i] for i in best_selected], dtype=np.int64)
    result_b = np.array(vectors_b, dtype=np.int64)
    
    final_vectors = np.vstack([result_a, result_b])
    
    return final_vectors
