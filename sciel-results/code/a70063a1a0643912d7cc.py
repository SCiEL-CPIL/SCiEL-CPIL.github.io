import time
import random

N = 12
SIZE = 1 << N
FULL_MASK = SIZE - 1

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton.
    """
    size = SIZE
    
    # Precompute transitions for all subsets
    # This is the bottleneck. For N=12, size=4096.
    # We can optimize by precomputing the image of every subset.
    # However, doing this from scratch every time is 4096 * (popcount) operations.
    # popcount avg 6. 4096 * 6 = 24k ops.
    # In Python, this is ~1-2ms.
    # If we do 1000 evaluations, that's 2 seconds.
    # With 55s budget, we can do ~25,000 evaluations.
    
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    # Optimization: Precompute bit indices for each state
    # Actually, just iterating bits is fast enough for N=12.
    
    for subset in range(1, size):
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = FULL_MASK
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Strategy: Enumerate a structured family of automata.
    # The "Pinwheel" automaton is a strong candidate for high reset thresholds.
    # Structure:
    # Letter 'a' is a permutation (often a single cycle of length N).
    # Letter 'b' is designed to reduce the number of states in the subset,
    # but "slowly".
    
    # A specific parameterized family:
    # Let 'a' be the cycle (0 -> 1 -> ... -> N-1 -> 0).
    # Let 'b' map i to i-1 (mod N) for most i, but with a "break" at some point.
    # Actually, the standard Pinwheel is:
    # a: i -> i+1 mod N
    # b: i -> i-1 mod N for i > 0, and b(0) = 0 (or similar).
    # But this might be too simple.
    
    # Let's try a more general "almost cycle" for 'a' and a "shift with break" for 'b'.
    
    # We will iterate through permutations of 'a' that are single cycles.
    # And variations of 'b'.
    
    # However, the space of single cycles is (N-1)! = 11! = 39,916,800.
    # Too many to iterate all with full BFS.
    
    # Instead, we fix 'a' to be the standard cycle (0->1->...->11->0).
    # Why? Because the relative order is what matters for the "pinwheel" effect.
    # Any single cycle is isomorphic to the standard cycle.
    # So we only need to search over 'b'.
    
    # The space of 'b' is 12^12 = 8.9 * 10^12. Still too big.
    
    # But we can restrict 'b' to be a "shift" with a few breaks.
    # b[i] = (i - 1 + N) % N is the standard reverse shift.
    # Let's start with b[i] = (i - 1) % N for all i.
    # Then we modify 1, 2, or 3 entries of 'b' to point to different states.
    
    # Let's also try variations where 'a' is not a single cycle but a permutation
    # with specific properties? No, single cycle is best for pinwheel.
    
    # Let's enumerate 'b' candidates.
    # Base b: i -> i-1 mod N.
    # We will try all possible single modifications to this base b.
    # There are N * N = 144 single modifications.
    # Then all possible double modifications.
    # C(N, 2) * N * N = 66 * 144 = 9504.
    # Then triple modifications.
    # C(N, 3) * N^3 = 220 * 1728 = 380k.
    # This is feasible.
    
    base_a = [(i + 1) % N for i in range(N)]
    base_b = [(i - 1) % N for i in range(N)]
    
    best_len = -1
    best_a = base_a[:]
    best_b = base_b[:]
    
    # Check the base case first
    l = compute_reset_length(base_a, base_b)
    if l > best_len:
        best_len = l
        best_a = base_a[:]
        best_b = base_b[:]
    
    # Enumerate single modifications to b
    for i in range(N):
        for target in range(N):
            if target == base_b[i]:
                continue
            b_mod = base_b[:]
            b_mod[i] = target
            l = compute_reset_length(base_a, b_mod)
            if l > best_len:
                best_len = l
                best_a = base_a[:]
                best_b = b_mod[:]
                
    if time.time() > deadline - 10:
        return [best_a, best_b]
        
    # Enumerate double modifications to b
    # To avoid duplicates and save time, we pick i < j
    for i in range(N):
        for j in range(i + 1, N):
            for target_i in range(N):
                if target_i == base_b[i]:
                    continue
                b_mod = base_b[:]
                b_mod[i] = target_i
                for target_j in range(N):
                    if target_j == base_b[j]:
                        continue
                    b_mod[j] = target_j
                    l = compute_reset_length(base_a, b_mod)
                    if l > best_len:
                        best_len = l
                        best_a = base_a[:]
                        best_b = b_mod[:]
                    b_mod[j] = base_b[j] # Restore for next target_j
            # Note: We can break early if time is tight, but double mods are fast enough.
            
    if time.time() > deadline - 5:
        return [best_a, best_b]
        
    # Enumerate triple modifications to b
    # This is 220 * 144 * 144 = ~4.5 million. Might be too slow in Python.
    # Let's sample or limit.
    # Actually, 4.5M * 2ms = 9000 seconds. Too slow.
    # We need to be smarter.
    
    # Alternative: Use a heuristic search on 'b' starting from the best found so far.
    # Or, restrict the targets for modifications to be "nearby" in the cycle.
    # In a pinwheel, the break should be at a specific location.
    # Usually, mapping a state to 0 or 1 is effective.
    
    # Let's try a local search from the best candidate found in double modifications.
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Randomized local search
    # Pick a random state i and random target t for b.
    # If it improves, keep it.
    
    random.seed(42) # Deterministic for reproducibility
    
    # We have ~50s left.
    # Each evaluation is ~2ms.
    # We can do ~25,000 evaluations.
    
    # Let's do a simple hill climbing.
    # But hill climbing often gets stuck.
    # Let's try "Random Restart Hill Climbing" or "Simulated Annealing".
    # Given the small space, let's just do a broader random search around the best structure.
    
    # Actually, let's try modifying 'a' as well?
    # If we fix 'b' to a good structure, maybe a different 'a' cycle is better?
    # No, all single cycles are isomorphic.
    
    # Let's try modifying 'b' with more random changes, not just 1-3.
    # Generate random 'b' that is "close" to the base_b.
    # e.g., 20% of entries are changed.
    
    attempts = 0
    while time.time() < deadline - 1:
        attempts += 1
        # Generate a candidate 'b'
        # Start with base_b
        b_cand = base_b[:]
        # Change k random positions
        k = random.randint(1, 4)
        indices = random.sample(range(N), k)
        for idx in indices:
            b_cand[idx] = random.randint(0, N - 1)
            
        l = compute_reset_length(base_a, b_cand)
        if l > best_len:
            best_len = l
            best_a = base_a[:]
            best_b = b_cand[:]
            
        # Also try modifying the best_b found so far
        if attempts % 10 == 0:
            b_cand2 = best_b[:]
            idx = random.randint(0, N - 1)
            b_cand2[idx] = random.randint(0, N - 1)
            l2 = compute_reset_length(base_a, b_cand2)
            if l2 > best_len:
                best_len = l2
                best_a = base_a[:]
                best_b = b_cand2[:]

    return [best_a, best_b]

# To ensure the function is available as required
if __name__ == "__main__":
    # Simple test
    a, b = construct()
    print(f"Best length: {compute_reset_length(a, b)}")
