import numpy as np
import time

_N = 256
_MAX_D = _N // 3 + 1

# Precompute all 4-term APs
_AP_LIST = []
# Map from index to list of AP indices
_INDEX_TO_APS = [[] for _ in range(_N)]

_ap_count = 0
for d in range(1, _MAX_D):
    for a in range(_N - 3 * d):
        i0 = a
        i1 = a + d
        i2 = a + 2 * d
        i3 = a + 3 * d
        
        _AP_LIST.append((i0, i1, i2, i3))
        _ap_count += 1
        
        _INDEX_TO_APS[i0].append(_ap_count - 1)
        _INDEX_TO_APS[i1].append(_ap_count - 1)
        _INDEX_TO_APS[i2].append(_ap_count - 1)
        _INDEX_TO_APS[i3].append(_ap_count - 1)

_N_APS = len(_AP_LIST)
_AP_ARRAY = np.array(_AP_LIST, dtype=np.int32)

def _thue_morse(n):
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

def _initial_score(arr):
    cols = arr[_AP_ARRAY]
    mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
    return np.sum(mono), mono

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_left():
        return deadline - time.time()

    # 1. Initialization
    # Thue-Morse is a strong start. Also try random restarts to escape bad basins.
    best_arr = _thue_morse(_N).copy()
    best_score, best_mono = _initial_score(best_arr)
    
    # Try a few random starts
    for _ in range(20):
        cand = np.random.randint(0, 2, _N, dtype=np.int8)
        s, m = _initial_score(cand)
        if s < best_score:
            best_score = s
            best_arr = cand.copy()
            best_mono = m.copy()

    # 2. Vectorized Greedy Coordinate Descent
    # Strategy: 
    # Repeatedly compute the delta score for flipping EVERY bit.
    # If any bit reduces the score, flip all such bits (or the best ones) simultaneously.
    # This is a "sweep". We repeat sweeps until no improvement is found or time runs out.
    
    cur_arr = best_arr.copy()
    cur_score = best_score
    cur_mono = best_mono.copy()
    
    # Precompute structure for vectorized delta calculation
    # For each AP j, we have 4 indices.
    # Let C be the color matrix (N_APS x 4).
    # An AP is mono if all 4 colors are equal.
    # Flipping bit i affects all APs containing i.
    
    # To vectorize the delta calculation for ALL i:
    # Delta_i = sum over j in APs(i) of (NewMono_j - OldMono_j)
    # OldMono_j is known (cur_mono).
    # NewMono_j depends on flipping i.
    # If AP j is currently mono:
    #   Flipping i (which is in j) makes it non-mono. Delta contribution = -1.
    # If AP j is currently non-mono:
    #   Flipping i makes it mono ONLY IF the other 3 bits in j are equal to (1 - cur_arr[i]).
    #   This is a specific condition.
    
    # Efficient approach:
    # 1. Identify all "bad" bits: bits that are part of at least one mono AP.
    #    Flipping these bits is guaranteed to reduce the score by at least 1 (for each mono AP they break).
    #    However, they might create new mono APs.
    # 2. A simpler heuristic that works well for this specific problem:
    #    "Greedy Break": Find all bits involved in mono APs. Flip them all.
    #    This might break many APs but create some new ones.
    #    Then, "Greedy Fix": Find bits that, if flipped, would fix the most newly created mono APs without breaking others.
    
    # Let's use a more robust vectorized gradient step.
    # Calculate delta for each i.
    # Delta_i = - (# of mono APs containing i) + (# of non-mono APs containing i that become mono if i flips)
    
    # Term 1: - (# of mono APs containing i)
    # We can compute this by summing cur_mono over APs for each i.
    # Since N_APS is small (~2000) and N=256, we can do this efficiently.
    
    # Let M be a (N_APS, 4) boolean matrix where M[j, k] = 1 if AP j is mono.
    # Actually, we just need to know for each i, how many mono APs it's in.
    # mono_count_per_bit = np.zeros(_N)
    # for j in range(_N_APS):
    #     if cur_mono[j]:
    #         for k in range(4):
    #             mono_count_per_bit[_AP_ARRAY[j, k]] += 1
    
    # This loop is O(N_APS * 4) which is ~8000 ops. Very fast.
    # But we can vectorize it:
    # mono_indices = np.where(cur_mono)[0]
    # mono_bits = _AP_ARRAY[mono_indices] # Shape (K, 4)
    # mono_count_per_bit = np.bincount(mono_bits.flatten(), minlength=_N)
    
    # Term 2: (# of non-mono APs containing i that become mono if i flips)
    # An AP j becomes mono if we flip i in it, iff:
    # 1. AP j is currently non-mono.
    # 2. The other 3 bits in AP j are equal to each other AND equal to (1 - cur_arr[i]).
    
    # Let's precompute for each AP j:
    # - Is it non-mono?
    # - Are any 3 bits equal?
    # - If so, what is the value of those 3 bits?
    # - Which position is the "odd one out" (if any)?
    
    # Actually, it's easier to just compute the delta for all i in a vectorized way using the AP array.
    # For each AP j, let colors be c0, c1, c2, c3.
    # If we flip c0, the new colors are 1-c0, c1, c2, c3.
    # This is mono if 1-c0 == c1 == c2 == c3.
    # This implies c1 == c2 == c3 AND c0 == 1-c1.
    
    # Let's create a matrix "would_be_mono_if_flip" of shape (N_APS, 4).
    # Entry [j, k] is 1 if flipping the k-th element of AP j makes it mono.
    
    # cols = cur_arr[_AP_ARRAY] # (N_APS, 4)
    # For each row j, we want to check:
    # Flip k: (cols[:, k] == 1 - cols[:, other1]) & (cols[:, other1] == cols[:, other2]) & ...
    
    # Let's do this for k=0,1,2,3.
    # Flip 0: (1 - cols[:,0]) == cols[:,1] == cols[:,2] == cols[:,3]
    # This is equivalent to: cols[:,1] == cols[:,2] == cols[:,3] AND cols[:,0] != cols[:,1]
    
    # Let's compute "3_equal" masks.
    # e01 = cols[:,0] == cols[:,1]
    # e02 = cols[:,0] == cols[:,2]
    # e03 = cols[:,0] == cols[:,3]
    # e12 = cols[:,1] == cols[:,2]
    # e13 = cols[:,1] == cols[:,3]
    # e23 = cols[:,2] == cols[:,3]
    
    # Flip 0 makes mono if: e12 & e13 & e23 & (~e01)  # The other 3 are equal, and c0 is different
    # Flip 1 makes mono if: e02 & e03 & e23 & (~e01)
    # Flip 2 makes mono if: e01 & e03 & e13 & (~e02)
    # Flip 3 makes mono if: e01 & e02 & e12 & (~e03)
    
    # Note: If all 4 are equal (mono), flipping any makes it non-mono.
    # Our formula for "Flip 0 makes mono" requires ~e01. If all 4 are equal, e01 is True, so ~e01 is False.
    # So "Flip 0 makes mono" is False for mono APs. Correct.
    # If 3 are equal and 1 is different, flipping the different one makes it mono.
    # Example: 0, 1, 1, 1. Flip 0 -> 1, 1, 1, 1. Mono.
    # e12 (1==1)=T, e13 (1==1)=T, e23 (1==1)=T. e01 (0==1)=F. ~e01=T. Result T. Correct.
    # Example: 0, 0, 1, 1. Flip 0 -> 1, 0, 1, 1. Not mono.
    # e12 (0==1)=F. Result F. Correct.
    
    # So, let's compute this matrix.
    
    improved = True
    while improved and time_left() > 2.0:
        improved = False
        
        cols = cur_arr[_AP_ARRAY]
        
        e01 = cols[:, 0] == cols[:, 1]
        e02 = cols[:, 0] == cols[:, 2]
        e03 = cols[:, 0] == cols[:, 3]
        e12 = cols[:, 1] == cols[:, 2]
        e13 = cols[:, 1] == cols[:, 3]
        e23 = cols[:, 2] == cols[:, 3]
        
        # Matrix of shape (N_APS, 4)
        # flip_mono[:, 0] is 1 if flipping pos 0 makes AP mono
        flip_mono = np.zeros((_N_APS, 4), dtype=np.int8)
        flip_mono[:, 0] = (e12 & e13 & e23 & ~e01)
        flip_mono[:, 1] = (e02 & e03 & e23 & ~e01)
        flip_mono[:, 2] = (e01 & e03 & e13 & ~e02)
        flip_mono[:, 3] = (e01 & e02 & e12 & ~e03)
        
        # Now, calculate delta for each bit i.
        # delta[i] = - (number of mono APs containing i) + (number of non-mono APs containing i that become mono if i flips)
        
        # Term 1: mono_count_per_bit
        mono_indices = np.where(cur_mono)[0]
        if len(mono_indices) > 0:
            mono_bits = _AP_ARRAY[mono_indices]
            mono_count_per_bit = np.bincount(mono_bits.flatten(), minlength=_N)
        else:
            mono_count_per_bit = np.zeros(_N, dtype=np.int8)
            
        # Term 2: gain_from_flip_per_bit
        # For each AP j, if flip_mono[j, k] is 1, then flipping bit at _AP_ARRAY[j, k] gains 1.
        # We need to sum this over all APs for each bit.
        # We can use bincount again.
        # Create a flat array of bits that would gain, and weight 1.
        # But we only want to count if the AP is currently NON-mono.
        # Our flip_mono calculation already ensures that the AP is NOT currently mono 
        # (because it requires ~e01 etc, which is false if all 4 are equal).
        # Wait, what if 2 are equal and 2 are different? e.g. 0,0,1,1.
        # Flip 0: 1,0,1,1. Not mono. e12(0==1)=F. Correct.
        # What if 0,1,0,1?
        # Flip 0: 1,1,0,1. Not mono.
        
        # So flip_mono correctly identifies "becomes mono".
        # We need to accumulate these gains to the corresponding bits.
        
        # Create a list of (bit_index, gain) pairs? No, vectorized bincount is better.
        # We need to map (j, k) -> bit_index = _AP_ARRAY[j, k]
        # And sum flip_mono[j, k] for each bit_index.
        
        # This is a sparse matrix multiplication or a scatter-add.
        # Since N_APS is small, we can do:
        # gains = np.zeros(_N, dtype=np.int8)
        # for k in range(4):
        #     bits_k = _AP_ARRAY[:, k]
        #     gains += np.bincount(bits_k, weights=flip_mono[:, k], minlength=_N)
        
        gains = np.zeros(_N, dtype=np.int8)
        for k in range(4):
            bits_k = _AP_ARRAY[:, k]
            # Only add gains where flip_mono is 1
            # bincount with weights handles this
            gains += np.bincount(bits_k, weights=flip_mono[:, k], minlength=_N)
            
        delta = gains - mono_count_per_bit
        
        # Find bits where delta < 0 (flipping reduces score)
        beneficial_bits = np.where(delta < 0)[0]
        
        if len(beneficial_bits) > 0:
            # Flip all beneficial bits
            cur_arr[beneficial_bits] = 1 - cur_arr[beneficial_bits]
            cur_score += np.sum(delta[beneficial_bits])
            
            # Update mono status
            # Recalculate mono status for all APs affected by these flips
            # For simplicity and correctness, just recalculate all mono statuses.
            # N_APS is ~2000, so this is fast.
            cols = cur_arr[_AP_ARRAY]
            cur_mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
            
            if cur_score < best_score:
                best_score = cur_score
                best_arr = cur_arr.copy()
                best_mono = cur_mono.copy()
            
            improved = True
        else:
            # No single bit flip reduces score.
            # Try a random perturbation to escape local optimum?
            # Or try flipping 2 bits?
            # Let's try a random restart from the current best if stuck.
            # Actually, let's try flipping a random bit even if delta > 0, with some probability?
            # No, let's stick to greedy. If stuck, try a different initial state.
            # Since we are in a loop, we can break and try a new restart.
            break

    # If we broke due to no improvement, we might be stuck in a local optimum.
    # Let's do a few random restarts from scratch if time permits.
    if time_left() > 5.0:
        for _ in range(10):
            if time_left() < 5.0: break
            start_arr = np.random.randint(0, 2, _N, dtype=np.int8)
            s, m = _initial_score(start_arr)
            
            # Run a quick greedy optimization on this start
            cur_arr = start_arr.copy()
            cur_score = s
            cur_mono = m.copy()
            
            local_improved = True
            # Limit steps per restart to ensure we finish
            steps = 0
            while local_improved and steps < 100 and time_left() > 5.0:
                local_improved = False
                steps += 1
                
                cols = cur_arr[_AP_ARRAY]
                e01 = cols[:, 0] == cols[:, 1]
                e02 = cols[:, 0] == cols[:, 2]
                e03 = cols[:, 0] == cols[:, 3]
                e12 = cols[:, 1] == cols[:, 2]
                e13 = cols[:, 1] == cols[:, 3]
                e23 = cols[:, 2] == cols[:, 3]
                
                flip_mono = np.zeros((_N_APS, 4), dtype=np.int8)
                flip_mono[:, 0] = (e12 & e13 & e23 & ~e01)
                flip_mono[:, 1] = (e02 & e03 & e23 & ~e01)
                flip_mono[:, 2] = (e01 & e03 & e13 & ~e02)
                flip_mono[:, 3] = (e01 & e02 & e12 & ~e03)
                
                mono_indices = np.where(cur_mono)[0]
                if len(mono_indices) > 0:
                    mono_bits = _AP_ARRAY[mono_indices]
                    mono_count_per_bit = np.bincount(mono_bits.flatten(), minlength=_N)
                else:
                    mono_count_per_bit = np.zeros(_N, dtype=np.int8)
                    
                gains = np.zeros(_N, dtype=np.int8)
                for k in range(4):
                    bits_k = _AP_ARRAY[:, k]
                    gains += np.bincount(bits_k, weights=flip_mono[:, k], minlength=_N)
                    
                delta = gains - mono_count_per_bit
                beneficial_bits = np.where(delta < 0)[0]
                
                if len(beneficial_bits) > 0:
                    cur_arr[beneficial_bits] = 1 - cur_arr[beneficial_bits]
                    cur_score += np.sum(delta[beneficial_bits])
                    cols = cur_arr[_AP_ARRAY]
                    cur_mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
                    local_improved = True
                    
            if cur_score < best_score:
                best_score = cur_score
                best_arr = cur_arr.copy()
                best_mono = cur_mono.copy()

    return best_arr.tolist()

# EVOLVE-BLOCK-END
