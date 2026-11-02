import time
import random
import numpy as np
from itertools import product

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    # To avoid circular convolution artifacts, we need size >= 2*N - 1? 
    # No, max sum is 2*mx = N-1. Max diff is mx.
    # Linear convolution of two length N signals has length 2N-1.
    # We only care about indices 0..2*mx for sum (which is 0..N-1).
    # For diff, we care about indices 0..2*mx in the reversed convolution?
    # Let's just do full linear convolution via FFT size 2N or next power of 2.
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    # rev_sig[i] = sig[N-1-i]
    # We want counts of a-b.
    # Standard trick: conv(sig, flip(sig))
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    # The index k in S_diff corresponds to a + (N-1-b) = k => a-b = k - (N-1).
    # We want to know if a-b exists.
    # a-b ranges from -mx to mx.
    # k ranges from 0 to 2*mx.
    # Let's check non-zero entries in S_diff.
    # Specifically, we want to map back to valid differences.
    # Since U contains 0, min diff is -mx, max diff is mx.
    # The convolution covers all combinations.
    # diffset_size = np.sum(S_diff > 1e-6) is safe because only valid diffs will have mass.
    
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

def construct_mixed_radix(bases, digits):
    """
    Construct a set U from mixed-radix parameters.
    bases: list of integers [b1, b2, ..., bk]
    digits: list of lists [[d1_min, d1_max], [d2_min, d2_max], ...]
    U = { sum(d_i * product(b_1...b_{i-1})) for d_i in digits_i }
    """
    U = [0]
    multiplier = 1
    for i in range(len(bases)):
        b = bases[i]
        d_min, d_max = digits[i]
        # Current U is the set of values for lower digits
        # New U = { u + d * multiplier for u in U, d in range(d_min, d_max+1) }
        new_U = []
        for u in U:
            for d in range(d_min, d_max + 1):
                new_U.append(u + d * multiplier)
        U = new_U
        multiplier *= b
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy: Search over mixed-radix structures.
    # A set defined by bases b_1..b_k and digit ranges [0, n_i-1]
    # has size product(n_i) and max element approx product(b_i).
    # Sumset size is roughly product(n_i + n_i - 1) if bases are large enough (no carries).
    # Diffset size is similar.
    # We want to maximize log(|U-U|/|U+U|) / log(2 max U + 1).
    
    # If we use symmetric digit ranges around 0, U-U and U+U might be similar.
    # But U is non-negative.
    
    # Let's try to optimize a parameterized family.
    # Family: U = { sum d_i * M_i } where M_i are powers of a base B.
    # This is just standard base-B representation with digits in [0, D_i].
    
    # Key insight: If we choose bases such that the sumsets don't overlap much (large gaps),
    # the sumset size is the product of the individual sumset sizes.
    # |U+U| = prod |D_i + D_i|
    # |U-U| = prod |D_i - D_i|
    # Ratio = prod (|D_i - D_i| / |D_i + D_i|)
    # Max U = sum (D_i * M_{i-1})
    
    # For a single digit set D = {0, ..., k}, |D+D| = 2k+1, |D-D| = 2k+1. Ratio = 1.
    # We need a digit set where |D-D| > |D+U|.
    # Example: D = {0, 1, 3}. |D+D|=6, |D-D|=7. Ratio 7/6.
    # If we have multiple levels, the total ratio is the product of level ratios.
    # The denominator is log(2 * MaxU + 1).
    # MaxU grows exponentially with the number of levels if bases are > 1.
    # So we want high ratio per level but small growth in MaxU.
    
    # Let's define a "block" structure.
    # We will search for a sequence of "blocks" (digit sets) and "bases" (separators).
    
    # Initial candidate blocks (small sets with ratio > 1)
    # We need to find small sets S containing 0 such that |S-S|/|S+S| > 1.
    # Let's pre-compute a few good small sets.
    
    good_blocks = []
    # Brute force small sets to find good "atoms"
    # A set S of size m.
    # We want to maximize r(S) = |S-S|/|S+S|.
    # And minimize span(S) relative to r(S).
    
    # Let's generate some candidates manually or via quick search.
    # S = {0, 1, 3} -> r=7/6=1.166, span=3
    # S = {0, 1, 4} -> S+S={0,1,2,4,5,8} (6), S-S={-4,-3,-1,0,1,3,4} (7). r=7/6. span=4.
    # S = {0, 2, 5} -> S+S={0,2,4,5,7,10} (6), S-S={-5,-3,-2,0,2,3,5} (7). r=7/6. span=5.
    # S = {0, 1, 2, 5} -> S+S={0..4, 5..7, 10} -> {0,1,2,3,4,5,6,7,10} (9).
    # S-S={-5..5} except -4,4? No.
    # 0,1,2,5.
    # Diffs:
    # 0: 0
    # 1: 1, -1
    # 2: 2, -2, 1, -1
    # 5: 5, -5, 4, -4, 3, -3
    # Unique: -5,-4,-3,-2,-1,0,1,2,3,4,5. Size 11.
    # Ratio 11/9 = 1.22. Span 5.
    
    # S = {0, 1, 2, 6}
    # S+S: 0,1,2,3,4, 6,7,8, 12. Size 9.
    # S-S: -6..6. Size 13.
    # Ratio 13/9 = 1.44. Span 6.
    
    # S = {0, 1, 3, 4}
    # S+S: 0,1,2,3,4,5,6,7,8. Size 9.
    # S-S: -4..4. Size 9. Ratio 1.
    
    # It seems "gapped" sets work better.
    # Let's use a search to find the best "atom" sets up to size 6 or 7.
    
    atoms = []
    # Generate all subsets of {0..10} of size 2..5 containing 0
    # This is 2^10 * 5 = ~5000 checks. Fast.
    max_span_atom = 15
    for size in range(2, 6):
        # Combinations of 1..max_span_atom of length size-1
        from itertools import combinations
        for comb in combinations(range(1, max_span_atom + 1), size - 1):
            S = [0] + list(comb)
            S.sort()
            # Calculate ratio
            # Use simple set ops for small sets
            sumset = set()
            diffset = set()
            for a in S:
                for b in S:
                    sumset.add(a + b)
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
            if r > 1.0:
                atoms.append((r, max(S), S))
                
    # Sort atoms by ratio descending, then span ascending
    atoms.sort(key=lambda x: (-x[0], x[1]))
    
    # Top atoms
    # We will build a set U by concatenating these atoms with large gaps (bases).
    # U = A_1 * B_1 + A_2 * B_2 * B_1 + ...
    # Actually, mixed radix:
    # U = { d_1 * 1 + d_2 * M_1 + d_3 * M_2 + ... }
    # where d_i in Atom_i.
    # M_i = product of previous "effective spans" to ensure no overlap in sumsets.
    # To ensure |U+U| = prod |A_i+A_i|, we need the span of the lower part times 2 to be less than the gap to the next part.
    # Specifically, if U_k = sum_{i=1}^k d_i M_{i-1},
    # Max sum of lower k-1 is 2 * Max(U_{k-1}).
    # Min element of next level is 0 * M_{k-1} = 0? No, d_k >= 0.
    # If we want disjoint sumsets, we need the support of the convolution to not overlap.
    # This is guaranteed if M_{k-1} > 2 * Max(U_{k-1}).
    # Let's enforce M_{k-1} = 2 * Max(U_{k-1}) + 1.
    
    # We want to maximize log(prod r_i) / log(2 * MaxU + 1)
    # = sum log(r_i) / log(2 * MaxU + 1)
    
    # Let's try to construct such a set using a greedy or local search on the sequence of atoms.
    
    best_sequence_score = 0.0
    best_sequence_atoms = []
    
    # We can use up to K atoms.
    # MaxU grows roughly exponentially.
    # If we pick an atom with ratio r and span s,
    # and we use it K times,
    # MaxU ~ s^K * (2K) roughly.
    # Score ~ K * log(r) / log(2 * s^K) ~ K * log(r) / (K * log(s)) = log(r) / log(s).
    # So the score converges to log(r)/log(s) for large K.
    # We want to maximize log(r)/log(s) for the atom.
    
    # Let's find the atom with best "efficiency" log(r)/log(s).
    best_eff_atom = None
    best_eff_val = 0.0
    for r, s, S in atoms:
        if s <= 1: continue
        eff = np.log(r) / np.log(s)
        if eff > best_eff_val:
            best_eff_val = eff
            best_eff_atom = S
            
    # Initialize with the best efficiency atom repeated
    # But we can mix atoms.
    
    # Local search on the sequence of atoms.
    # State: list of atoms (indices into `atoms` list).
    # Start with a sequence of the best atom.
    
    max_depth = 20 # Don't go too deep, MaxU gets huge and FFT fails.
    
    current_seq = [best_eff_atom] * 5
    current_score = 0.0
    
    # Calculate score for a sequence
    def eval_sequence(seq):
        U = [0]
        M = 1
        for atom in seq:
            # Expand U with this atom at scale M
            new_U = []
            for u in U:
                for d in atom:
                    new_U.append(u + d * M)
            U = new_U
            # Update M for next level
            # To prevent overlap in sumsets, we need M_next > 2 * Max(U)
            # But we are defining M_next based on the current U's max.
            # Wait, the "base" for the next digit is determined by the span of the current U.
            # Let M_next = 2 * max(U) + 1.
            if not U: return 0, 0, 0
            M = 2 * max(U) + 1
            # Check size
            if len(U) > 100000: # FFT limit
                return 0, 0, 0
        if not U: return 0, 0, 0
        mx = max(U)
        if mx > 10**6: # Too large for FFT in reasonable time
            return 0, 0, 0
        s = _score_set(U)
        return s, mx, len(U)
    
    s, mx, n = eval_sequence(current_seq)
    if s > best_sequence_score:
        best_sequence_score = s
        best_sequence_atoms = current_seq[:]
        best_set = construct_mixed_radix_from_seq(current_seq) # Need to return actual set
        
    # Actually, let's just keep the best set found in eval_sequence.
    # Re-implement eval to return the set.
    
    def eval_sequence_full(seq):
        U = [0]
        M = 1
        for atom in seq:
            new_U = []
            for u in U:
                for d in atom:
                    new_U.append(u + d * M)
            U = new_U
            if not U: return 0.0, []
            M = 2 * max(U) + 1
            if len(U) > 50000: # Safety
                return 0.0, []
        if not U: return 0.0, []
        mx = max(U)
        if mx > 10**6:
            return 0.0, []
        s = _score_set(U)
        return s, U
        
    s, U = eval_sequence_full(current_seq)
    if s > best_score:
        best_score = s
        best_set = U
        
    # Local search on the sequence
    # Moves:
    # 1. Replace an atom with another atom from `atoms`
    # 2. Add an atom at the end
    # 3. Remove an atom from the end
    
    time_limit = deadline
    while time.time() < time_limit:
        improved = False
        
        # 1. Replace
        if len(current_seq) > 0:
            idx = random.randint(0, len(current_seq) - 1)
            # Pick a random atom from the top 50 atoms
            if atoms:
                new_atom = atoms[random.randint(0, min(50, len(atoms))-1)][2]
                new_seq = current_seq[:]
                new_seq[idx] = new_atom
                s, U = eval_sequence_full(new_seq)
                if s > best_score:
                    best_score = s
                    best_set = U
                    current_seq = new_seq
                    improved = True
        
        # 2. Add
        if len(current_seq) < max_depth and atoms:
            new_atom = atoms[random.randint(0, min(50, len(atoms))-1)][2]
            new_seq = current_seq + [new_atom]
            s, U = eval_sequence_full(new_seq)
            if s > best_score:
                best_score = s
                best_set = U
                current_seq = new_seq
                improved = True
        
        # 3. Remove
        if len(current_seq) > 1:
            new_seq = current_seq[:-1]
            s, U = eval_sequence_full(new_seq)
            if s > best_score:
                best_score = s
                best_set = U
                current_seq = new_seq
                improved = True
        
        # 4. Random Restart / Perturbation
        if not improved and random.random() < 0.05:
             # Try a random short sequence
             L = random.randint(2, 5)
             current_seq = [atoms[random.randint(0, min(50, len(atoms))-1)][2] for _ in range(L)]
             s, U = eval_sequence_full(current_seq)
             if s > best_score:
                 best_score = s
                 best_set = U

    return best_set

# Helper to construct from seq (used in thought process, but eval_sequence_full returns U directly)
def construct_mixed_radix_from_seq(seq):
    U = [0]
    M = 1
    for atom in seq:
        new_U = []
        for u in U:
            for d in atom:
                new_U.append(u + d * M)
        U = new_U
        M = 2 * max(U) + 1
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
