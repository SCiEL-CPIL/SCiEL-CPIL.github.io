"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary sequence of length N with exactly N/2 ones.
    We minimize the maximum circular cross-correlation between h and 1-h.
    
    The overlap at shift k is: sum_i h[i] * (1-h[(i+k) % N])
    = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - autocorr[k]
    
    where autocorr[k] = sum_i h[i]*h[(i+k)%N]
    
    So maximizing the minimum autocorr[k] for k != 0 minimizes the max overlap.
    Equivalently, we want to maximize min_{k!=0} autocorr[k].
    
    Since sum h = N/2 = 50, and h is binary, autocorr[0] = 50.
    For k != 0, we want autocorr[k] to be as large as possible (to minimize overlap).
    
    We use an FFT-based method to compute autocorrelation quickly, and a local search
    with swaps to optimize the sequence.
    """
    
    # Initialize with an alternating pattern for a good start
    h = np.zeros(N)
    h[0::2] = 1.0
    
    def get_autocorr(h):
        """
        Compute circular autocorrelation using FFT.
        autocorr[k] = sum_i h[i] * h[(i+k) % N]
        This is the circular cross-correlation of h with h.
        """
        # Pad to power of 2 for FFT efficiency, though N=100 is small
        # We need circular correlation, so zero-padding is NOT used for the convolution itself
        # if we want circular. But standard np.correlate with 'full' and slicing works.
        # Alternatively, use FFT:
        # C[k] = sum_i h[i] h[i+k]
        # This is equivalent to the inverse FFT of |FFT(h)|^2 if we handle the indexing correctly.
        # Let's stick to a direct O(N^2) check but optimized with numpy dot products
        # because N=100 is very small. O(N^2) = 10,000 ops. 
        # The previous code was slow because it called objective() inside the swap loop 
        # which recomputed everything. 
        # Let's just compute the full autocorr vector once per candidate.
        
        # Using FFT for circular autocorrelation:
        # The circular autocorrelation r[k] is the inverse DFT of |H|^2.
        # r[k] = IDFT( H * conj(H) )
        # Note: numpy.fft.ifft returns complex, we take real part.
        # The index k in numpy fft corresponds to shift.
        # We need to be careful with the definition.
        # np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h)))
        # gives the circular cross-correlation.
        
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        return autocorr

    def get_max_overlap(h):
        """
        Compute the maximum overlap over all non-zero shifts.
        overlap[k] = N/2 - autocorr[k]
        max_overlap = max(overlap[k] for k in 1..N-1)
        """
        autocorr = get_autocorr(h)
        # We exclude k=0 because overlap[0] = N/2 - N/2 = 0? 
        # Wait. overlap[0] = sum h[i](1-h[i]) = 0 for binary.
        # The task asks for overlap with shifted 1-h.
        # Usually "shifted" implies non-zero shift, or all shifts?
        # "minimize the maximum overlap of h with shifted 1-h"
        # If shift 0 is included, overlap is 0.
        # If shift k != 0, we look at k=1..N-1.
        # The prompt says "shifted", implying k != 0.
        # Also, the score is "upper bound on C5".
        # Let's assume k=1..N-1.
        
        # autocorr[0] is N/2.
        # We want to minimize max_{k!=0} (N/2 - autocorr[k])
        # which is N/2 - min_{k!=0} autocorr[k]
        
        min_autocorr = np.min(autocorr[1:])
        return N / 2.0 - min_autocorr

    best_h = h.copy()
    best_score = get_max_overlap(h)
    
    # Local Search: Swap a 1 and a 0
    # We try all pairs (i, j) where h[i]=1 and h[j]=0.
    # To speed up, we can precompute the effect of a swap on the autocorrelation.
    # However, N=100 is small enough that a simple loop with fast objective might work.
    # Let's do a hill-climbing with random restarts.
    
    np.random.seed(42)
    
    for restart in range(50):
        if restart == 0:
            h = best_h.copy()
        else:
            # Random restart
            h = np.zeros(N)
            indices = np.random.choice(N, N//2, replace=False)
            h[indices] = 1.0
        
        current_score = get_max_overlap(h)
        
        improved = True
        # Limit iterations per restart to avoid infinite loops
        max_iters = 1000
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Find all 1s and 0s
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            
            # To avoid O(N^2) full recomputation for every pair,
            # we can just iterate through a random subset or all pairs.
            # With N=100, 50*50 = 2500 pairs. 
            # 2500 * 50 restarts * 1000 iters = 125 million operations? 
            # No, the inner loop is just checking if score improves.
            # get_max_overlap is O(N log N) or O(N^2).
            # Let's optimize by only trying a few random swaps per iteration if it's slow.
            # But N=100 is tiny. Let's try all pairs but break early if found.
            
            # Shuffle to avoid bias
            ones = np.random.permutation(ones)
            zeros = np.random.permutation(zeros)
            
            found_improvement = False
            for i in ones:
                for j in zeros:
                    # Swap h[i] and h[j]
                    h[i] = 0.0
                    h[j] = 1.0
                    
                    new_score = get_max_overlap(h)
                    
                    if new_score < current_score:
                        current_score = new_score
                        found_improvement = True
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_h = h.copy()
                        # Break out of loops to restart search from new state
                        break
                    else:
                        # Revert
                        h[i] = 1.0
                        h[j] = 0.0
                
                if found_improvement:
                    break
            
            if not found_improvement:
                # No improvement found in this full pass?
                # Actually, we broke early on first improvement.
                # If we finished the loops without finding any, improved is False.
                # But we set improved=True if we found one.
                # So if we exit the for loops without finding one, improved remains False.
                pass

    return best_h
# EVOLVE-BLOCK-END
