import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build inverse mapping: for each index i, which APs does it participate in?
    # And what is its position (0, 1, 2, 3) in those APs?
    ap_indices_per_pos = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        a0, a1, a2, a3 = ap_array[ap_idx]
        ap_indices_per_pos[a0].append((ap_idx, 0))
        ap_indices_per_pos[a1].append((ap_idx, 1))
        ap_indices_per_pos[a2].append((ap_idx, 2))
        ap_indices_per_pos[a3].append((ap_idx, 3))
        
    # Precompute other indices for each AP and position
    # other_indices[ap_idx][pos] = tuple of the other 3 indices
    other_indices = [[None]*4 for _ in range(num_aps)]
    for ap_idx in range(num_aps):
        a0, a1, a2, a3 = ap_array[ap_idx]
        other_indices[ap_idx][0] = (a1, a2, a3)
        other_indices[ap_idx][1] = (a0, a2, a3)
        other_indices[ap_idx][2] = (a0, a1, a3)
        other_indices[ap_idx][3] = (a0, a1, a2)
        
    def compute_is_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        return (c0 == c1) & (c1 == c2) & (c2 == c3)
        
    def local_search_sa(color_vec, is_mono, current_score, deadline):
        """
        Simulated Annealing local search with exact delta evaluation.
        Returns updated color_vec, is_mono, score.
        """
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        best_score = current_score
        best_color = color_vec.copy()
        best_is_mono = is_mono.copy()
        
        while time.time() < deadline - 0.5 and T > min_T:
            # Shuffle order of indices
            indices = np.random.permutation(n)
            
            for idx in indices:
                if time.time() > deadline - 0.5:
                    break
                    
                # Calculate delta if we flip color_vec[idx]
                delta = 0
                for ap_idx, pos in ap_indices_per_pos[idx]:
                    if is_mono[ap_idx]:
                        delta -= 1
                    else:
                        other_idx_0, other_idx_1, other_idx_2 = other_indices[ap_idx][pos]
                        c0 = color_vec[other_idx_0]
                        c1 = color_vec[other_idx_1]
                        c2 = color_vec[other_idx_2]
                        
                        if c0 == c1 and c1 == c2:
                            if c0 != color_vec[idx]:
                                delta += 1
                
                # Acceptance criterion
                if delta < 0:
                    accept = True
                else:
                    # Metropolis criterion
                    if T > 0:
                        accept_prob = np.exp(-delta / T)
                        accept = (random.random() < accept_prob)
                    else:
                        accept = False
                        
                if accept:
                    color_vec[idx] = 1 - color_vec[idx]
                    current_score += delta
                    
                    # Update is_mono for affected APs
                    for ap_idx, pos in ap_indices_per_pos[idx]:
                        a0, a1, a2, a3 = ap_array[ap_idx]
                        c0 = color_vec[a0]
                        c1 = color_vec[a1]
                        c2 = color_vec[a2]
                        c3 = color_vec[a3]
                        is_mono[ap_idx] = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_color = color_vec.copy()
                        best_is_mono = is_mono.copy()
                        
                    if best_score == 0:
                        return best_color, best_is_mono, best_score
                        
            T *= cooling_rate
            
        return best_color, best_is_mono, best_score

    best_score = float('inf')
    best_color = None
    
    # Multi-start strategy
    while time.time() < deadline - 1.0:
        # Choose initialization
        init_type = random.randint(0, 3)
        
        if init_type == 0:
            # Thue-Morse
            color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        elif init_type == 1:
            # Random
            color = np.random.randint(0, 2, n, dtype=np.int8)
        elif init_type == 2:
            # Random with bias
            color = np.zeros(n, dtype=np.int8)
            for _ in range(10):
                start = random.randint(0, n-1)
                length = random.randint(1, 10)
                color[start:start+length] = 1
        else:
            # Alternating blocks of 2
            color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
            
        is_mono = compute_is_mono(color)
        current_score = np.sum(is_mono)
        
        # Local Search with Simulated Annealing
        color, is_mono, current_score = local_search_sa(color, is_mono, current_score, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()
            
        if best_score == 0:
            break
            
        # Perturbation for next iteration
        if best_color is not None and time.time() < deadline - 2.0:
            perturbed = best_color.copy()
            num_flips = random.randint(5, 10)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            is_mono_p = compute_is_mono(perturbed)
            score_p = np.sum(is_mono_p)
            
            perturbed, is_mono_p, score_p = local_search_sa(perturbed, is_mono_p, score_p, deadline)
            
            if score_p < best_score:
                best_score = score_p
                best_color = perturbed.copy()
                
                if best_score == 0:
                    break

    if best_color is None:
        best_color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
        
    return best_color.tolist()

# EVOLVE-BLOCK-END
