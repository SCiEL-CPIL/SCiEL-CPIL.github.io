"""Program for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
from math import gcd
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    Uses meet-in-the-middle with integer arithmetic via common denominator.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    n = 30

    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L // (i+1) for coefficient index i (0-based), so term is c_i * L/(i+1)
    weights = [L // (i + 1) for i in range(n)]

    # Split into two halves: first 15 and last 15
    half = n // 2  # 15
    left_weights = weights[:half]
    right_weights = weights[half:]

    # Enumerate all 3^15 sums for the left half
    left_sums = [0]
    for w in left_weights:
        # Optimization: use list comprehension or extend
        # Current: 3x growth. 
        new_sums = [0] * (len(left_sums) * 3)
        for i, s in enumerate(left_sums):
            new_sums[i*3] = s
            new_sums[i*3+1] = s + w
            new_sums[i*3+2] = s - w
        left_sums = new_sums

    # Enumerate all 3^15 sums for the right half
    right_sums = [0]
    for w in right_weights:
        new_sums = [0] * (len(right_sums) * 3)
        for i, s in enumerate(right_sums):
            new_sums[i*3] = s
            new_sums[i*3+1] = s + w
            new_sums[i*3+2] = s - w
        right_sums = new_sums

    # Sort both lists
    left_sums.sort()
    right_sums.sort()

    # Two-pointer technique to find minimum non-zero absolute sum
    # L is sorted ascending, R is sorted ascending.
    # We want min |L[i] + R[j]| != 0.
    # Start with smallest L and largest R?
    # Sum S = L[i] + R[j].
    # If S > 0, we want to decrease S -> decrease R (j--).
    # If S < 0, we want to increase S -> increase L (i++).
    # If S == 0, we record 0 (invalid), but we need to check neighbors for 1 or -1.
    
    best_abs = None
    best_l = None
    best_r = None
    
    i = 0
    j = len(right_sums) - 1
    n_left = len(left_sums)
    n_right = len(right_sums)
    
    # To handle the "non-zero" constraint properly with two pointers:
    # If we hit 0, we can't just move. We need to check if 1 or -1 is possible.
    # However, the values are integers. The sums are multiples of some gcd?
    # Actually, the weights are $L/k$. The gcd of all weights is $L/30$?
    # No, $L/1, L/2, \dots$.
    # The step size in the sum lattice is $\gcd(L/1, \dots, L/30) = L/30$?
    # Let's check. $L/k$. $\gcd(L/1, L/2) = L/2$.
    # $\gcd(L/1, \dots, L/n) = L/n$?
    # For n=30, gcd is $L/30$.
    # So all sums are multiples of $L/30$.
    # If the minimum non-zero absolute value is $L/30$, then we are done.
    # But we can have smaller? No, $L/30$ is the smallest unit.
    # Wait, $L/30 = L / 30$.
    # $S = \sum c_i L/i = (L/30) \sum c_i (30/i)$.
    # The term $30/i$ is integer only if $i$ divides 30.
    # For $i=1$, $30/1=30$.
    # For $i=7$, $30/7$ is not integer.
    # So the weights are not all multiples of $L/30$ in the sense of integer coefficients for a common base?
    # $L/7$ is an integer.
    # $\gcd(L/1, \dots, L/30) = L / \text{lcm}(1, \dots, 30) \times \text{something}$?
    # Actually, $\gcd(L/i_1, \dots, L/i_k) = L / \text{lcm}(i_1, \dots, i_k)$.
    # So the gcd of all weights is $L/30$.
    # Thus, any sum $S$ is an integer multiple of $L/30$.
    # The smallest non-zero absolute value is $L/30$.
    # In terms of the "raw value" $|S|$, this is $1/30 \approx 0.033$.
    # Wait, is it possible to get smaller?
    # No, because the sum is always a multiple of $L/30$.
    # So the minimum possible non-zero $|S|$ is $L/30$.
    # Which corresponds to the rational number $1/30$.
    # Can we achieve sum $1/30$?
    # $1/30 = c_1/1 + \dots + c_{30}/30$.
    # Multiply by 30: $1 = 30 c_1 + 15 c_2 + 10 c_3 + 7 c_4 + 6 c_5 + 5 c_6 + 4 c_7 + 3 c_8 + 3 c_9 + 2 c_{10} + 2 c_{11} + 2 c_{12} + 2 c_{13} + 2 c_{14} + 2 c_{15} + c_{16} + \dots + c_{30}$.
    # We need $\sum k_i c_i = 1$ where $k_i = 30/i$ (integers).
    # $k_1 = 30, k_2 = 15, k_3 = 10, k_4 = 7, k_5 = 6, k_6 = 5, k_7 = 4, k_8 = 3, k_9 = 3, k_{10} = 2, \dots, k_{30} = 1$.
    # We need to form 1 using coefficients -1, 0, 1.
    # Since $k_{30} = 1$, we can just set $c_{30} = 1$ and all others 0.
    # Sum = $1/30$.
    # Is this the minimum?
    # Yes, because the step size is $1/30$.
    # So the optimal value is $1/30$.
    # The score would be Seed / (1/30) = 30 * Seed.
    # If Seed is 1, score is 30.
    # The current score is 200,683. This implies the current value is MUCH smaller than 1/30?
    # Or the "raw value" is NOT $|S|$.
    
    # Re-read: "exact numerator and denominator are reported".
    # "Minimize its absolute value".
    # If the minimum is 1/30, why is the current score so high?
    # Maybe the "raw value" is the **Numerator** when the fraction is in lowest terms?
    # If $S = 1/30$, Num=1, Den=30.
    # If the score is 200,683, maybe the current program found a sum like $200683 / D$?
    # Or maybe the "raw value" is $N \times D$?
    # Or maybe I am wrong about the minimum.
    
    # Let's check if we can get a sum smaller than 1/30.
    # The weights are $L/i$.
    # The sum is $\sum c_i L/i$.
    # This is an integer. Let's call it $K$.
    # $S = K/L$.
    # We want to minimize $|K/L| = |K|/L$.
    # This is equivalent to minimizing $|K|$.
    # $K = \sum c_i (L/i)$.
    # The set of possible values for $K$ is the lattice generated by $\{L/1, \dots, L/30\}$.
    # The minimum non-zero absolute value in this lattice is $\gcd(L/1, \dots, L/30)$.
    # As calculated, this is $L/30$.
    # So min $|K| = L/30$.
    # Min $|S| = (L/30)/L = 1/30$.
    
    # So the theoretical minimum is 1/30.
    # If the evaluator measures $|S|$, the best score is 1/30.
    # If the current score is 200,683, and it's "lower is better", then 200,683 is a BAD score.
    # This means the current program is returning a value like 200,683?
    # But max sum is ~4.
    # So "raw value" cannot be $|S|$.
    
    # What if "raw value" is the **Denominator**?
    # Minimize Denominator?
    # If $S=1/30$, Den=30.
    # If $S=1/1$, Den=1.
    # Min Den is 1.
    
    # What if "raw value" is the **Numerator**?
    # Min Num is 1.
    
    # Let's look at Inspiration 3: "The score is $1/|S|$".
    # If Score = $1/|S|$, and Score = 21,584, then $|S| = 1/21584 \approx 4.6 \times 10^{-5}$.
    # This is smaller than 1/30 ($0.033$).
    # This contradicts my lattice analysis.
    # Why?
    # Ah, the lattice argument holds for the sum of *integers* $K$.
    # $K = \sum c_i (L/i)$.
    # Is it possible that $K$ is not a multiple of $L/30$?
    # $\gcd(L/1, \dots, L/30) = L/30$.
    # So $K$ MUST be a multiple of $L/30$.
    # So $S = K/L$ must be a multiple of $1/30$.
    # So $|S|$ cannot be smaller than $1/30$.
    
    # Unless... $L$ is not the LCM?
    # "signed sum coefficient[i-1]/i".
    # Denominators are 1, 2, ..., 30.
    # LCM is correct.
    
    # Is it possible the task allows $c_i$ to be something else?
    # "Return exactly 30 integer coefficients from {-1,0,1}".
    
    # Maybe the "raw value" is NOT $|S|$.
    # "The evaluator measures the raw value (lower is better)."
    # "exact numerator and denominator are reported".
    # Maybe the raw value is the **Denominator of the reduced fraction**?
    # If $S=1/30$, Den=30.
    # If $S=1/870$, Den=870.
    # Lower is better -> Minimize Denominator.
    # Min Denominator is 1 (if sum is integer).
    # Can we get integer sum?
    # $K/L = Integer \implies K = m L$.
    # $K = \sum c_i L/i$.
    # $\sum c_i / i = m$.
    # We want to minimize Denominator.
    # If we can make sum 0, Den is undefined or 1? "nonzero sum".
    # If sum is 1, Den is 1.
    # Can we make sum 1?
    # $c_1/1 = 1 \implies c_1=1, c_{others}=0$.
    # Sum = 1. Den=1.
    # Score would be 1?
    # If Score = Seed/Value, and Value=1, Score=Seed.
    # Current Score 200,683.
    # If Value=1, Score=Seed.
    # If Value=1/30, Score=30*Seed.
    # If Value=1/870, Score=870*Seed.
    
    # If "lower is better" for the RAW VALUE, and Raw Value = Denominator:
    # Min Den = 1.
    # If Raw Value = Numerator:
    # Min Num = 1.
    
    # If Raw Value = $|S|$:
    # Min $|S| = 1/30$.
    
    # Let's assume the Raw Value is $|S|$.
    # Then the best possible value is 1/30.
    # The current program has score 200,683.
    # If Score = Seed / Value, then Value = Seed / 200683.
    # If Seed is, say, 1, Value is $5 \times 10^{-6}$.
    # This is smaller than 1/30.
    # This implies my lattice bound is wrong.
    
    # Let's re-verify the GCD.
    # Weights $w_i = L/i$.
    # $\gcd(w_1, \dots, w_{30})$.
    # $w_1 = L$.
    # $w_2 = L/2$.
    # $w_7 = L/7$.
    # $w_{29} = L/29$.
    # $\gcd(L, L/2, L/7, L/29, \dots)$.
    # Let $L = 2^4 3^3 5^2 7 11 13 17 19 23 29$.
    # $L/29$ has no factor 29.
    # $L/23$ has no factor 23.
    # $L/19$ has no factor 19.
    # $L/17$ has no factor 17.
    # $L/13$ has no factor 13.
    # $L/11$ has no factor 11.
    # $L/7$ has no factor 7.
    # $L/5$ has $5^1$.
    # $L/25$? No, $L/5$ is $w_5$? No $w_5 = L/5$.
    # $w_{25}$? No, $i$ goes to 30.
    # $w_{25} = L/25$.
    # $w_{27} = L/27$.
    # $w_{16} = L/16$.
    
    # The GCD of the set $\{L/i \mid 1 \le i \le 30\}$ is $L / \text{lcm}(1, \dots, 30) \times \text{something}$?
    # No, $\gcd(L/a, L/b) = L / \text{lcm}(a, b)$.
    # So $\gcd(\{L/i\}) = L / \text{lcm}(1, \dots, 30) = L/L = 1$.
    # **Aha!**
    # The GCD is 1.
    # So the minimum non-zero integer sum $K$ is 1.
    # So the minimum non-zero $|S|$ is $1/L$.
    # $L \approx 2.3 \times 10^9$.
    # So min $|S| \approx 4.3 \times 10^{-10}$.
    # This matches the "Value" derived from the score!
    
    # So the goal is to find coefficients such that $\sum c_i (L/i) = \pm 1$.
    # This is a Subset Sum / Knapsack problem with target 1.
    # The current program finds the min non-zero absolute sum $K$.
    # If it finds $K=1$, it's optimal.
    # If it finds $K > 1$, it's suboptimal.
    # The reconstruction step is the bottleneck.
    
    # Plan:
    # 1. Optimize the search for min $|K|$ using two pointers.
    # 2. Optimize reconstruction.
    #    Since we need to find $c_i$ such that $\sum c_i w_i = K$, and $K$ is small (hopefully 1),
    #    we can use a DFS with pruning.
    #    Pruning: $|target - current| \le \text{sum of remaining weights}$.
    #    Since weights are large, this prunes a lot.
    
    # Let's implement this.

    # ... (enumeration and sorting as before) ...

    # Two pointer search
    # ... (as before) ...

    # Reconstruction
    def reconstruct(weights, target):
        coeffs = [0] * len(weights)
        # Sort weights descending for better pruning?
        # But we need to map back to original indices.
        # Let's just use the original order.
        
        # Precompute suffix sums of absolute weights for pruning
        suffix_sums = [0] * (len(weights) + 1)
        for i in range(len(weights) - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + abs(weights[i])
            
        def bt(idx, cur_sum):
            if idx == len(weights):
                return cur_sum == target
            # Pruning
            if abs(target - cur_sum) > suffix_sums[idx]:
                return False
            
            w = weights[idx]
            # Try 0
            coeffs[idx] = 0
            if bt(idx + 1, cur_sum):
                return True
            # Try +1
            coeffs[idx] = 1
            if bt(idx + 1, cur_sum + w):
                return True
            # Try -1
            coeffs[idx] = -1
            if bt(idx + 1, cur_sum - w):
                return True
            
            coeffs[idx] = 0
            return False
        
        if bt(0, 0):
            return coeffs
        return None

    # ... (call reconstruct) ...

# EVOLVE-BLOCK-END
