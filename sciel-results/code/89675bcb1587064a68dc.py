"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        
        # Precompute sphere indices: for each word index, the 17 indices it covers
        # Using base-3 representation
        spheres = np.zeros((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            # Decode idx to base-3 digits
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()
            
            spheres[idx, 0] = idx
            pos = 1
            for pos_idx in range(n):
                original = digits[pos_idx]
                for d in range(q):
                    if d != original:
                        new_idx = idx + (d - original) * (3 ** (n - 1 - pos_idx))
                        spheres[idx, pos] = new_idx
                        pos += 1
        
        # Greedy set cover
        uncovered = np.ones(total_space, dtype=bool)
        code_indices = []
        
        while True:
            if time.time() > deadline - 40:
                break
            
            # Count how many uncovered words each candidate covers
            counts = uncovered[spheres].sum(axis=1)
            best_count = int(counts.max())
            
            if best_count <= 0:
                break
            
            best_idx = int(np.argmax(counts))
            code_indices.append(best_idx)
            uncovered[spheres[best_idx]] = False
        
        # Convert to active mask
        m = len(code_indices)
        active = np.ones(m, dtype=bool)
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[spheres[idx]] += 1
        
        # Local search: remove redundant codewords
        sphere_idx_matrix = spheres[code_indices]  # (m, 17)
        rng = random.Random(42)
        
        for pass_num in range(2000):
            if time.time() > deadline - 5:
                break
            
            if not np.any(active):
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # Find redundant codewords (all their sphere words have cover_count >= 2)
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            
            if len(redundant_local_indices) == 0:
                break
            
            # Randomize removal order
            if len(redundant_local_indices) > 1:
                order = list(range(len(redundant_local_indices)))
                rng.shuffle(order)
                redundant_local_indices = redundant_local_indices[order]
            
            removed_any = False
            for li in redundant_local_indices:
                if time.time() > deadline - 5:
                    break
                gi = active_indices[li]
                active[gi] = False
                cover_count[spheres[code_indices[gi]]] -= 1
                removed_any = True
            
            if not removed_any:
                break
        
        # Pairwise exchange local search: try to swap codewords to enable further reductions
        if time.time() < deadline - 10:
            for swap_round in range(500):
                if time.time() > deadline - 5:
                    break
                
                active_indices = np.where(active)[0]
                if len(active_indices) < 2:
                    break
                
                # Find codewords with unique coverage (min_cover == 1)
                active_spheres = sphere_idx_matrix[active_indices]
                active_cover_counts = cover_count[active_spheres]
                min_covers = np.min(active_cover_counts, axis=1)
                
                critical_local = np.where(min_covers == 1)[0]
                if len(critical_local) == 0:
                    break
                
                found_improvement = False
                for ci in critical_local:
                    if time.time() > deadline - 5:
                        break
                    
                    gi = active_indices[ci]
                    gi_sphere = spheres[code_indices[gi]]
                    
                    # Find words covered only by this codeword
                    unique_words = gi_sphere[cover_count[gi_sphere] == 1]
                    if len(unique_words) == 0:
                        continue
                    
                    # Try to find a replacement codeword
                    inactive_indices = np.where(~active)[0]
                    if len(inactive_indices) == 0:
                        break
                    
                    # Sample candidates
                    if len(inactive_indices) > 100:
                        sample = rng.sample(list(inactive_indices), 100)
                    else:
                        sample = list(inactive_indices)
                    
                    for si in sample:
                        if time.time() > deadline - 5:
                            break
                        
                        si_sphere = spheres[code_indices[si]]
                        
                        # Check if si covers all unique words of gi
                        si_covered = np.zeros(len(unique_words), dtype=bool)
                        for j, uw in enumerate(unique_words):
                            si_covered[j] = np.any(si_sphere == uw)
                        
                        if not np.all(si_covered):
                            continue
                        
                        # Try the swap
                        old_cover = cover_count.copy()
                        cover_count[gi_sphere] -= 1
                        cover_count[si_sphere] += 1
                        
                        # Check if any other active codeword becomes redundant
                        new_active_indices = [x for x in active_indices if x != gi]
                        if len(new_active_indices) > 0:
                            new_spheres = sphere_idx_matrix[new_active_indices]
                            new_cover_counts = cover_count[new_spheres]
                            new_min_covers = np.min(new_cover_counts, axis=1)
                            new_redundant = np.where(new_min_covers >= 2)[0]
                            
                            if len(new_redundant) > 0:
                                # Perform swap and remove redundant
                                active[gi] = False
                                active[si] = True
                                
                                ri = new_active_indices[new_redundant[0]]
                                active[ri] = False
                                cover_count[spheres[code_indices[ri]]] -= 1
                                
                                found_improvement = True
                                break
                        
                        cover_count = old_cover
                        if found_improvement:
                            break
                    
                    if found_improvement:
                        break
                
                if not found_improvement:
                    break
        
        # Final: remove any remaining redundant codewords
        for pass_num in range(500):
            if time.time() > deadline - 2:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            
            if len(redundant_local_indices) == 0:
                break
            
            for li in redundant_local_indices:
                gi = active_indices[li]
                active[gi] = False
                cover_count[spheres[code_indices[gi]]] -= 1
        
        # Build final code
        final_indices = [code_indices[i] for i in range(m) if active[i]]
        
        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[spheres[idx]] = True
        
        if np.all(verify_cover):
            # Convert indices back to words
            final_code = []
            for idx in final_indices:
                digits = []
                tmp = idx
                for _ in range(n):
                    digits.append(tmp % 3)
                    tmp //= 3
                digits.reverse()
                final_code.append(digits)
            return final_code
        
        # Fallback: return greedy code
        code = []
        for idx in code_indices:
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()
            code.append(digits)
        return code
    
    else:
        # General case
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
