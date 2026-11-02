import time
import numpy as np
from itertools import product

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if len(arr) == 0:
        return 0.0
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # We construct U using a mixed-radix representation with a specific structure.
    # U = { sum_{i=0}^{K-1} d_i B^i } where d_i in D.
    # To maximize the ratio |U-U|/|U+U|, we want the digit set D to have a high ratio.
    # For a set D, if we choose D = {0, s, 2s, ..., (L-1)s}, it's an arithmetic progression.
    # |D+D| = 2L-1, |D-D| = 2L-1. Ratio is 1. This is bad for the log term (contributes 0).
    # We need |D-D| > |D+D|.
    # Consider D = {0, 1, ..., L-1} \cup {M, M+1, ...}.
    # Actually, the current program found that "atoms" with r > 1 exist.
    # Let's look at the structure of sets with r > 1.
    # If D is "sparse" enough, collisions in sums are less likely than in differences?
    # No, differences are always at least as many as sums for symmetric sets?
    # Wait, for a set D containing 0, |D-D| >= |D+D| is not always true?
    # Example: D={0,1,2}. D+D={0,1,2,3,4} (size 5). D-D={-2,-1,0,1,2} (size 5). Ratio 1.
    # D={0,1,3}. D+D={0,1,2,3,4,6} (size 6). D-D={-3,-2,-1,0,1,2,3} (size 7). Ratio 7/6 > 1.
    # So non-arithmetic progressions can have ratio > 1.
    
    # The previous program searched for small atoms.
    # This new strategy:
    # 1. Define a parametric family of "good" digit sets based on skipping elements in an AP.
    #    e.g., D = {0, 1, ..., L-1} with some gaps.
    #    Specifically, let's try D = {0, 1, ..., L-1} \ {some elements} or D = {0, s, 2s, ...} is bad.
    #    Let's try D = {0, 1, ..., L-1} where we remove elements to break the AP structure?
    #    Actually, removing elements from a full AP {0..L-1} usually reduces the ratio towards 1 or below?
    #    {0,1,3} is {0,1,2,3} \ {2}. Ratio 7/6.
    #    {0,1,2,4}: D+D={0,1,2,3,4,5,6,8} (8). D-D={-4..4} (9). Ratio 9/8.
    #    {0,1,2,5}: D+D={0,1,2,3,4,5,6,7,10} (9). D-D={-5..5} (11). Ratio 11/9.
    
    # Let's use a greedy construction of the base B and the digit set D.
    # We want to maximize the contribution of each level.
    # Score contribution of one level with digit set D and base B (where B > 2*max(D)):
    # The total max U grows as M_k = sum d_i B^i.
    # If we use the SAME digit set D and SAME base B for K levels:
    # U is the set of numbers in base B with digits in D.
    # |U| = |D|^K.
    # max(U) = (B^K - 1)/(B - 1) * max(D) (approx, actually sum max(D) B^i).
    # Let M_max = max(D).
    # Max(U) = M_max * (B^K - 1)/(B - 1).
    # Denominator of score: log(2 * Max(U) + 1) approx K * log(B) + log(M_max).
    # Numerator of score: K * log(|D-D|/|D+D|).
    # Score approx 1 + K * log(r_D) / (K * log(B) + log(M_max)).
    # As K -> infinity, Score -> 1 + log(r_D) / log(B).
    # To maximize this, we want to maximize log(r_D) / log(B).
    # This means we want a digit set D with high ratio r_D, and a small base B.
    # Constraint: B > 2 * max(D) to ensure the mixed radix property holds (no carries/overlaps).
    # So B_min = 2 * max(D) + 1.
    # We want to minimize B, so we choose B = 2 * max(D) + 1.
    # Then we want to maximize log(r_D) / log(2 * max(D) + 1).
    
    # Let's search for the best digit set D of small size (up to 10-15 elements)
    # and compute the efficiency eff(D) = log(r_D) / log(2 * max(D) + 1).
    # Then we use that D and B for many levels K.
    # We also need to choose K to balance the finite-K effects and the size of U.
    # Size of U = |D|^K. We should keep |U| manageable (e.g., < 100,000).
    
    best_eff = -np.inf
    best_D = [0, 1, 3] # Default
    best_B = 7
    best_r = 7/6
    
    # Search space for D:
    # D must contain 0.
    # Let max_D be the largest element.
    # We can represent D by its elements.
    # To keep search fast, we'll iterate over possible max_D and sizes.
    
    max_max_D = 20
    max_size_D = 12
    
    # Precompute r_D for all subsets of {1, ..., max_max_D} of size up to max_size_D-1
    # This is still large: C(20, 11) is ~350k. C(20, 10) is ~184k.
    # We can prune: we only care about sets where r_D > 1.
    # Also, we can fix the structure: D = {0} U S where S is a subset of {1, ..., max_max_D}.
    
    # Let's do a smarter search.
    # We know that "gaps" help.
    # Try D = {0, 1, ..., L-1} with some holes.
    # Or just brute force small sets.
    
    import itertools
    
    # We will search for D in the range [0, max_max_D].
    # To speed up, we'll only consider sets where max(D) is the actual max.
    
    candidates = []
    
    # Iterate over possible max elements M
    for M in range(2, max_max_D + 1):
        # D must contain 0 and M.
        # Other elements are from {1, ..., M-1}.
        # We want to maximize r_D / log(2M+1).
        # Let's try a few heuristics for D.
        
        # Heuristic 1: D = {0, 1, ..., M-1, M} (Full AP) -> r=1. Skip.
        # Heuristic 2: D = {0, M} -> r=1. Skip.
        # Heuristic 3: D = {0, 1, M}
        D_test = [0, 1, M]
        # Calculate r
        sumset = set()
        diffset = set()
        for a in D_test:
            for b in D_test:
                sumset.add(a+b)
                diffset.add(a-b)
        r = len(diffset) / len(sumset)
        if r > 1:
            eff = np.log(r) / np.log(2 * M + 1)
            candidates.append((eff, D_test, M, r))
            
        # Heuristic 4: D = {0, 1, 2, M}
        D_test = [0, 1, 2, M]
        sumset = set()
        diffset = set()
        for a in D_test:
            for b in D_test:
                sumset.add(a+b)
                diffset.add(a-b)
        r = len(diffset) / len(sumset)
        if r > 1:
            eff = np.log(r) / np.log(2 * M + 1)
            candidates.append((eff, D_test, M, r))
            
        # Heuristic 5: D = {0, 1, M-1, M}
        D_test = [0, 1, M-1, M]
        sumset = set()
        diffset = set()
        for a in D_test:
            for b in D_test:
                sumset.add(a+b)
                diffset.add(a-b)
        r = len(diffset) / len(sumset)
        if r > 1:
            eff = np.log(r) / np.log(2 * M + 1)
            candidates.append((eff, D_test, M, r))

        # Heuristic 6: Random subsets of {1, ..., M-1} of size 1 or 2, plus 0, M
        if M > 2:
            for subset in itertools.combinations(range(1, M), 1):
                D_test = [0, M] + list(subset)
                D_test.sort()
                sumset = set()
                diffset = set()
                for a in D_test:
                    for b in D_test:
                        sumset.add(a+b)
                        diffset.add(a-b)
                r = len(diffset) / len(sumset)
                if r > 1:
                    eff = np.log(r) / np.log(2 * M + 1)
                    candidates.append((eff, D_test, M, r))
            
            for subset in itertools.combinations(range(1, M), 2):
                D_test = [0, M] + list(subset)
                D_test.sort()
                sumset = set()
                diffset = set()
                for a in D_test:
                    for b in D_test:
                        sumset.add(a+b)
                        diffset.add(a-b)
                r = len(diffset) / len(sumset)
                if r > 1:
                    eff = np.log(r) / np.log(2 * M + 1)
                    candidates.append((eff, D_test, M, r))
                    
    if not candidates:
        # Fallback
        return [0, 1, 3]
        
    # Sort by efficiency
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_eff, best_D, best_M, best_r = candidates[0]
    best_B = 2 * best_M + 1
    
    # Now determine K (number of levels)
    # We want |U| = |best_D|^K to be reasonable.
    # Let's try K such that |U| is between 1000 and 50000.
    # Also, we should check if mixing different D's helps?
    # Theoretically, using the single best D is optimal for large K.
    # For small K, maybe a different D is better due to the +1 in the score formula?
    # Score = 1 + K * log(r) / (K * log(B) + log(M_max)).
    # Let's just pick K to maximize the actual score for a few candidate K values.
    
    best_K = 1
    best_final_score = -np.inf
    best_U = best_D[:] # K=1
    
    # Try K from 1 to 10
    for K in range(1, 11):
        size_U = len(best_D) ** K
        if size_U > 100000:
            break
            
        # Construct U
        U = [0]
        B = best_B
        for _ in range(K):
            new_U = []
            for u in U:
                for d in best_D:
                    new_U.append(u + d * B)
            U = new_U
            B *= best_B # Wait, B is the base for the next digit.
            # In mixed radix, M_{i+1} = 2 * Max(U_i) + 1.
            # If we use fixed base B, we need B > 2 * max(D).
            # Max(U_i) = sum_{j=0}^{i} max(D) B^j = max(D) * (B^{i+1}-1)/(B-1).
            # Next base B_{i+1} must be > 2 * Max(U_i).
            # If we fix B, we need B > 2 * Max(U_i) for all i.
            # This grows exponentially. So we can't use a fixed base B for many levels
            # unless B is huge.
            # The "Mixed Radix" approach in the previous program used M_{i+1} = 2 Max(U_i) + 1.
            # This is the tightest possible base.
            # If we use the tightest base, the base grows.
            # Let's re-evaluate the "Efficiency" with growing bases.
            
            # If M_{i+1} = 2 Max(U_i) + 1, and Max(U_i) approx Max(D) * M_i * (B-1)/B?
            # Actually, if D is fixed, Max(U_i) = Max(D) * (M_i - 1) / (B-1) * B? No.
            # Let's just simulate the growth.
            
        # The simulation above was flawed because B changes.
        # Let's simulate the exact growth for the chosen D.
        
        U_sim = [0]
        M_curr = 0
        valid = True
        for _ in range(K):
            new_U = []
            for u in U_sim:
                for d in best_D:
                    new_U.append(u + d * M_curr)
            U_sim = new_U
            if not U_sim:
                valid = False
                break
            M_curr = 2 * max(U_sim) + 1
            if M_curr > 10**18:
                valid = False
                break
            if len(U_sim) > 100000:
                valid = False
                break
                
        if not valid:
            continue
            
        # Calculate score
        score = _score_set(U_sim)
        if score > best_final_score:
            best_final_score = score
            best_K = K
            best_U = U_sim[:]
            
    if not best_U:
        return [0, 1, 3]
        
    return best_U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
