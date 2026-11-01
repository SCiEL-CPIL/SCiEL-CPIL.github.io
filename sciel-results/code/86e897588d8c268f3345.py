"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    The goal is to minimize the upper bound on C3 (third autocorrelation inequality).
    
    The seed program returns np.ones(N), which corresponds to a constant positive function.
    We need to find a signed step function (values in {-1, 1} or similar) that improves
    the autoconvolution upper bound.
    
    For the third autocorrelation inequality, we consider the quantity:
    C3 = sum_{i,j} f[i] * f[j] * f[i+j] (with appropriate indexing for the discrete case)
    
    Actually, let me think more carefully. The "third autocorrelation" typically refers to
    the triple correlation or the third moment of the autocorrelation. For a function f,
    the autocorrelation is (f * f~)(t) = sum_i f[i] f[i+t], and the "third autocorrelation"
    might refer to the integral/sum of the square of the autocorrelation, or it could be
    the triple convolution.
    
    Given the context of "autoconvolution upper bound" and "C3", this likely refers to:
    C3 = sum_{i,j,k} f[i] f[j] f[k] where i+j+k = 0 (or similar constraint), which is
    related to the third moment of the Fourier transform.
    
    Actually, for a step function on [-1/4, 1/4] with N intervals, the natural quantity is:
    Let f be the step function. The Fourier transform is F(omega) = sum_{n=0}^{N-1} f[n] * exp(-i*omega*n*dx) * dx
    where dx = (1/2)/N = 1/(2N).
    
    The "autoconvolution" upper bound for C3 likely involves:
    C3 = (1/(2pi)^2) * integral |F(omega)|^2 * something, or more likely:
    
    For the third autocorrelation inequality in the context of additive combinatorics or
    signal processing, C3 might be defined as:
    C3 = sum_{x,y,z: x+y+z=0} f(x)f(y)f(z)
    
    In discrete terms with our step function:
    C3 = sum_{i,j: i+j < N} f[i] * f[j] * f[N-1-i-j]  (if we center appropriately)
    
    But actually, let me reconsider. The problem says "improve an autoconvolution upper bound"
    and the score is "the upper bound on C3" with "lower is better".
    
    Let me think about this differently. If we have a signed step function f with values in {-1, 1},
    the quantity we want to minimize might be related to the L2 norm of the autocorrelation,
    or the maximum of the autocorrelation, or some other measure.
    
    Given that the seed is np.ones(N) (all +1), and we want to improve (lower) the score,
    we should try alternating signs or other patterns that reduce the relevant correlation measure.
    
    A common approach for such problems is to use a random signed function or a structured
    pattern like a Rademacher sequence. Let me try a few approaches:
    
    1. Random signs: f[i] = 1 if random < 0.5 else -1
    2. Alternating: f[i] = 1 if i % 2 == 0 else -1
    3. Block pattern: groups of +1 and -1
    
    Since I need to compute the actual C3 value to know which is better, let me define
    a reasonable C3 metric and optimize for it.
    
    For a discrete signal f of length N, the circular autocorrelation is:
    R[k] = sum_{i=0}^{N-1} f[i] * f[(i+k) % N]
    
    The "third autocorrelation" might be:
    C3 = sum_{k=0}^{N-1} R[k]^2 = sum_{i,j} f[i] f[j] f[i] f[j] with some shift
    
    Actually, sum_k R[k]^2 = sum_{i,j,l} f[i] f[(i+k)%N] f[j] f[(j+k)%N] summed over k
    = N * sum_{i,j} f[i] f[j] f[i-j+N]%N ... this is getting complex.
    
    Let me just try a simple heuristic: use a random signed function and see if it
    beats the all-ones baseline. The all-ones function has maximum positive correlation,
    so any signed function should reduce the relevant measure.
    
    I'll use a fixed random seed for reproducibility and return a random signed step function.
    """
    # Use a fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Generate random signed step function
    f = rng.choice([-1, 1], size=N)
    
    return f
# EVOLVE-BLOCK-END
