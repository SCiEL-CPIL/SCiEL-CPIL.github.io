"""
Strategy: RBF Neural Network with Internal Gradient Descent
Instead of a fixed global basis, we use a Radial Basis Function (RBF) model.
The 10 parameters are the weights for 10 fixed Gaussian basis functions.
The centers and widths are fixed to cover the domain [-3, 3] with varying scales.
To improve fit quality beyond a simple linear combination (which BFGS might struggle with if the basis is ill-conditioned or if the function is complex),
the equation function performs a few steps of internal gradient descent (L-BFGS-like via numerical gradients or just simple SGD) 
to refine the parameters starting from the provided 'params' (which are initialized to ones by the evaluator).
However, since the evaluator calls minimize on 'equation', and 'equation' returns predictions,
we can actually just return the linear combination but choose a better conditioned basis.
Wait, the evaluator fits params. If I do internal optimization, I am effectively changing the loss landscape.
Actually, a simpler structural change is to use a **Fourier Series** with many harmonics, but 10 params is small.
Let's stick to the RBF idea but ensure the basis is well-conditioned.
The previous program used a mix of poly/trig/gauss.
The inspiration 3 (score 124) used a GMM with learnable centers/widths? No, it said "fixed, equally spaced" centers.
Let's try a **Pure RBF** with centers spread across [-3, 3] and widths chosen to be well-conditioned.
To make it structurally different and potentially better, I will use a **Multi-scale RBF** basis.
Centers: -3, -2, -1, 0, 1, 2, 3, and some intermediate.
Widths: A mix of narrow and wide to capture both local and global features.
Since the evaluator uses BFGS on the training data, the quality of the fit depends on the condition number of the design matrix.
Polynomials of degree 9 are ill-conditioned.
RBFs with fixed centers are often better conditioned.

Let's define 10 Gaussian basis functions:
Centers: np.linspace(-3, 3, 10)
Widths: A constant width might be too rigid. Let's use a shared width parameter? No, params are fixed length 10.
So the 10 params must be the amplitudes.
The centers and widths are hardcoded.
To maximize flexibility, I will use a "bump" basis that is orthogonal-ish or at least well-spread.

Actually, looking at the inspirations, p21 (Gaussian Basis) scored 1.039 (worse than 1.0 baseline? No, 1.0 is seed. 1.039 is worse. 124 is much better).
The seed score is 7020684434 (huge, meaning it failed or was terrible).
The baseline "1.0" in the prompt implies the seed program's score is normalized to 1.0?
"1.0 equals the seed program".
Seed program score: 7020684434.1069.
So if I get a score of 7020684434, my ratio is 1.0.
Inspiration 3 score: 124.5507. This is a HUGE improvement (factor of ~56 million).
This suggests the current seed is failing completely (maybe numerical instability or wrong shape).
The seed program looks syntactically correct. Why such a high score?
Maybe the target function is very complex and the linear combination of 10 terms is insufficient, leading to high MSE.
Or maybe the BFGS optimization fails.

Let's look at Inspiration 3's strategy: "Locally Adaptive Gaussian Basis... 10 fixed, equally spaced centers".
It scored 124.
The current program uses a mix of poly/trig.
I will implement a **Pure RBF** with 10 fixed centers and fixed widths, but I will choose the widths to be adaptive in a structural way (e.g., wider in the middle, narrower at edges) or just uniform.
To be structurally different from the current program (which has poly/trig), I will use **only Gaussians**.
I will also add a **constant term** explicitly? No, Gaussians at center 0 can act as a constant if wide enough.
Better: Use 9 Gaussians + 1 Linear term? No, keep it pure RBF for simplicity and conditioning.

Let's try **5 Gaussians** with **learnable widths**? No, params are fixed 10.
If I use 10 Gaussians, I have 10 amplitudes.
Centers: c_i = -3 + 6*i/9 for i=0..9.
Widths: sigma_i. If I fix sigma, the model is a linear RBF.
Is a linear RBF with 10 nodes enough to beat the seed?
The seed is a linear combination of 10 diverse functions.
If the seed scores 7e9, it's likely failing to converge or the basis is bad for the specific target.
A pure RBF basis is often more stable for local approximation.

Let's implement a **Multi-Scale RBF** where the widths vary.
Widths: [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0] ?
Or just uniform width 1.0.
Let's try uniform width 1.0 first, it's a standard RBF.

To improve on the "fixed basis" approach, I can try to make the basis **orthogonal** or use **Chebyshev polynomials** which are better conditioned than standard powers.
The current program uses standard powers x, x^2, x^3.
Chebyshev polynomials T_n(x) are bounded in [-1, 1] and orthogonal.
Scaling x to [-1, 1] and using Chebyshev up to degree 9 might be much better conditioned than raw powers.
This is a structural change in the basis representation.

Let's try **Chebyshev Polynomials** of the first kind.
x_scaled = x / 3.0
T_0 = 1
T_1 = x_scaled
T_2 = 2*x_scaled^2 - 1
...
T_9.
This avoids the ill-conditioning of high-degree monomials.
The current program has x, x^2, x^3, sin, cos, sin2, cos2, exp, 1/(1+x^2).
It's a mix.
I will replace the entire basis with **10 Chebyshev Polynomials** scaled to [-1, 1].
This is a different design family (orthogonal polynomial basis vs mixed basis).

Implementation:
1. Scale x to [-1, 1]: t = x / 3.0
2. Compute T_0(t) ... T_9(t)
3. y = sum(p[i] * T_i(t))

This is simple, robust, and well-conditioned.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 10 Chebyshev polynomials.
    The input x is scaled to [-1, 1] to ensure the Chebyshev polynomials are well-conditioned.
    This basis is orthogonal and avoids the numerical instability of high-degree monomials.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    
    # Scale x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0] = 1.0
    if len(x) > 0:
        T[1] = t
        for n in range(2, 10):
            T[n] = 2.0 * t * T[n-1] - T[n-2]
            
    # Linear combination
    # y = p[0]*T_0 + p[1]*T_1 + ... + p[9]*T_9
    y = np.sum(T * p[:, np.newaxis], axis=0)
    
    return y

# EVOLVE-BLOCK-END
