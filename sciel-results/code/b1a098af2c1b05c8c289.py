import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    best_seq = np.ones(N, dtype=np.int8)
    best_score = float('inf')
    
    def calc_score(seq):
        # seq is int8 array of 1/-1
        I = np.sum(seq)
        if I == 0:
            return float('inf')
        # Compute autocorrelation (linear)
        # Using FFT for speed? N=400 is small, direct is O(N^2)=160k ops.
        # FFT is O(N log N).
        # Let's use direct for simplicity and incremental updates later.
        R = np.correlate(seq, seq, mode='full')
        # R is length 2N-1. Peak is at N-1.
        # Off-peak are all others.
        # We want max abs of off-peak.
        R_offpeak = np.concatenate([R[:N-1], R[N:]])
        M = np.max(np.abs(R_offpeak))
        return M / (I * I)

    # Initial evaluation
    best_score = calc_score(best_seq)
    
    # Local search
    # We will try to improve by flipping signs.
    # To handle the "mostly +1" bias, we start with ones.
    # We can also try random initializations but weighted towards +1?
    # The metric J = M/I^2 rewards large I.
    # So we should stay close to all-ones.
    
    # Incremental update function
    def update_correlation(seq, R, idx):
        # Flip seq[idx]
        old_val = seq[idx]
        new_val = -old_val
        seq[idx] = new_val
        
        # Update R
        # R[k] = sum_{j} seq[j] seq[j+k]
        # Indices in 'full' correlate:
        # R[0] corresponds to shift N-1 (leftmost)
        # R[N-1] is peak
        # R[2N-2] is rightmost
        # Actually, np.correlate(a, v, 'full') returns c_k = sum_j a_{j+k} v_j.
        # For autocorrelation, a=v.
        # The peak is at index N-1.
        
        # Changes to R:
        # For a shift k (0 to 2N-2 in array index):
        # The terms involving idx are:
        # 1. j = idx, so a_{idx+k} v_{idx}
        # 2. v_j = v_{idx}, so j = idx, a_{j+k} = a_{idx+k}... wait.
        # c_k = sum_j a_{j+k} a_j.
        # Term 1: a_{idx+k} a_idx. Valid if 0 <= idx+k < N.
        # Term 2: a_idx a_{idx-k}. Valid if 0 <= idx-k < N.
        
        # In np.correlate output:
        # Index i in R corresponds to shift.
        # Let's just recompute R for simplicity if N is small, or use FFT.
        # N=400. R calculation is 160k mults.
        # 100s / (160k ops) is huge.
        # But Python loop overhead is high.
        # Let's use FFT for global recompute if local search stalls, 
        # or just use numpy correlate which is C-optimized.
        # np.correlate is fast enough for N=400.
        pass

    # Simpler approach: Use numpy correlate in the loop.
    # It's O(N^2) but C speed. 400^2 = 160,000.
    # 1e6 operations per second in Python? No, numpy is 1e8+.
    # So ~1000 evaluations per second?
    # 100s -> 100,000 evaluations.
    # Enough for hill climbing.

    def get_score_fast(seq):
        I = np.sum(seq)
        if I == 0: return float('inf')
        R = np.correlate(seq, seq, mode='full')
        R_offpeak = np.concatenate([R[:N-1], R[N:]])
        M = np.max(np.abs(R_offpeak))
        return M / (I * I)

    # Hill Climbing
    current_seq = best_seq.copy()
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each index
        # Randomize order to avoid local minima bias
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > deadline:
                break
            
            # Flip
            old_val = current_seq[i]
            current_seq[i] = -old_val
            
            new_score = get_score_fast(current_seq)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                # Update best
                if current_score < best_score:
                    best_score = current_score
                    best_seq = current_seq.copy()
            else:
                # Revert
                current_seq[i] = old_val
        
        # If no improvement, try a random restart with bias?
        # Or just break?
        # Let's try a few random restarts with "mostly +1"
        if not improved:
             # Try a random sequence with 80% +1
             if time.time() < deadline:
                 rand_seq = np.random.choice([1, -1], N, p=[0.8, 0.2]).astype(np.int8)
                 rand_score = get_score_fast(rand_seq)
                 if rand_score < best_score:
                     best_score = rand_score
                     best_seq = rand_seq.copy()
                     current_seq = rand_seq.copy()
                     current_score = rand_score
                     improved = True # Restart loop
                 else:
                     # Break if we've been stuck for a while?
                     # Just break to save time
                     break

    return best_seq

# EVOLVE-BLOCK-END
