"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # The task: minimize max|autoconvolution| / (integral)^2 over signed step functions.
    # Seed: ones(N) -> value 1/N (assuming k=0 is included in max, or k!=0 but ones is optimal).
    # Strategy: Represent signed step function as binary sequence mapped to {-1, 1} (or continuous heights).
    # The inspiration suggests discrete gradient descent on sign patterns.
    # However, the seed program argues that ones(N) is the global optimum for non-negative f,
    # and any signed perturbation increases sum f^2 relative to (sum f)^2, thus increasing the ratio.
    # Let's verify this with a quick local search using the discrete heuristic from the inspiration.
    
    # Start with the seed solution
    f = np.ones(N)
    
    # Time budget: 150s limit, use 140s for safety.
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define scoring function
    # C(k) = sum_{i} f[i] f[i+k]
    # Score = max_k |C(k)| / (sum f)^2
    # Note: The task says "maximum absolute autoconvolution". 
    # Usually autocorrelation includes k=0. If k=0 is included, C(0) = sum f^2.
    # For f=ones(N), C(0)=N, C(k!=0)=N-k. Max is N. Sum f = N. Ratio = N/N^2 = 1/N.
    # If k=0 is excluded, Max is N-1. Ratio = (N-1)/N^2.
    # The prompt says "signed autocorrelation maximum absolute-value variant".
    # Let's assume standard definition includes k=0 unless specified otherwise.
    # The seed comment says: "max autocorr = sum f^2 ... ratio >= 1/N". This implies k=0 is included.
    
    def calc_score(f):
        S = np.sum(f)
        if S == 0:
            return float('inf')
        # Compute autocorrelation via FFT
        # Zero-pad to 2*N to avoid circular convolution
        F = np.fft.fft(f, 2*N)
        C = np.fft.ifft(F * np.conj(F)).real[:N]
        max_corr = np.max(np.abs(C))
        return max_corr / (S * S)
    
    current_score = calc_score(f)
    best_f = f.copy()
    best_score = current_score
    
    # Discrete Gradient Descent Heuristic
    # We will try flipping signs of individual elements or small groups.
    # Since f is continuous in the problem statement ("signed step heights"), 
    # but the inspiration suggests binary {-1, 1}, let's consider that the optimal 
    # solution might lie on the boundary or be continuous.
    # However, the seed argues ones(N) is optimal. 
    # Let's try to improve by perturbing values slightly or flipping signs.
    # If we flip a sign in ones(N), S becomes N-2, sum f^2 remains N.
    # C(0) = N. Ratio = N / (N-2)^2.
    # For N=400, 1/N = 0.0025. N/(N-2)^2 = 400/398^2 = 400/158404 = 0.002525...
    # This is worse (higher). So flipping signs in ones(N) is bad.
    
    # What if we change magnitudes?
    # Let f[i] = 1 + epsilon_i.
    # This is a continuous optimization.
    # The objective is smooth except at points where the max changes.
    # We can use a simple local search on continuous values.
    
    # Let's try a random restart local search on continuous heights.
    # Initialize with ones.
    # Step: pick random indices, adjust values slightly.
    
    # To respect the "discrete gradient descent" inspiration while keeping the seed's insight:
    # We will perform a local search that allows continuous adjustments, 
    # but we will also test discrete sign flips to ensure we aren't stuck in a local minimum 
    # that a continuous search might miss (though for this convex-like problem, it's unlikely).
    
    # Actually, the most efficient check is to see if any perturbation lowers the score.
    # Given the strong theoretical argument for ones(N), we will spend most time verifying 
    # or doing a very tight local search around ones(N).
    
    # Let's do a simple simulated annealing or random walk on continuous values.
    # f[i] can be any real number.
    # But the problem says "signed step heights", implying the function is a step function 
    # with constant values over intervals. The array f represents these heights.
    
    # We will use a simple local search:
    # 1. Compute score.
    # 2. Pick a random index i.
    # 3. Propose new value f[i] = f[i] + delta.
    # 4. If score improves, accept.
    # 5. Repeat until time limit.
    
    # To be efficient, we can use the FFT-based score calculation.
    # Each evaluation is O(N log N). N=400 is small.
    # We can do many evaluations.
    
    # Let's also try the discrete binary search from the inspiration as a secondary path.
    # If the continuous search doesn't improve, try binary.
    
    # Hybrid approach:
    # Phase 1: Continuous local search around ones(N).
    # Phase 2: If no improvement, try discrete sign flips (binary search).
    
    # Phase 1: Continuous
    # We'll use a simple random step.
    # Step size should be adaptive.
    
    # Actually, let's stick to the discrete heuristic from the inspiration but applied to 
    # a continuous space? No, the inspiration says "binary sequence mapped to {-1, 1}".
    # Let's try that directly.
    
    # Start with ones(N) as the best known.
    # Try binary patterns.
    # Random binary pattern: f = 2 * (np.random.rand(N) > 0.5) - 1
    # This gives values in {-1, 1}.
    
    # Let's run a few random binary searches.
    
    # First, let's see if any random binary pattern beats ones(N).
    # Score of ones(N) is 1/N = 0.0025.
    # For a random binary pattern, sum f is close to 0 (if balanced).
    # If sum f is 0, score is inf.
    # We need sum f to be large.
    # So we need mostly +1s.
    # This is a constrained optimization: maximize sum f subject to max |C(k)| / (sum f)^2 being small.
    
    # Let's try a local search starting from ones(N) in the discrete space {-1, 1}.
    # But ones(N) is not in {-1, 1}. 
    # The problem allows "signed step heights", so continuous values are allowed.
    # The inspiration's strategy is just one idea. The seed's idea is stronger theoretically.
    # Let's combine: Start with ones(N). Do a local search that allows continuous changes.
    # If stuck, try discrete flips.
    
    # Let's implement a simple local search for continuous f.
    
    # We'll use a step size that decays.
    # Initial step size: 0.1
    # Decay: 0.99
    
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random index
        idx = np.random.randint(N)
        # Propose a new value
        delta = np.random.uniform(-step_size, step_size)
        old_val = f[idx]
        f[idx] = old_val + delta
        
        new_score = calc_score(f)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
            no_improve_count = 0
            # Increase step size slightly on improvement?
            # step_size *= 1.1
        else:
            # Revert
            f[idx] = old_val
            no_improve_count += 1
            
            # Decay step size if no improvement for a while
            if no_improve_count > 100:
                step_size *= 0.9
                no_improve_count = 0
                # If step size is very small, try discrete flips?
                if step_size < 1e-4:
                    # Try a discrete flip
                    f[idx] = -f[idx]
                    new_score = calc_score(f)
                    if new_score < best_score:
                        best_score = new_score
                        best_f = f.copy()
                    else:
                        f[idx] = -f[idx] # revert
    
    return best_f

# EVOLVE-BLOCK-END
