import time
import random

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 50

def _init_bitsets():
    """
    Initialize bitsets for all 2^N inputs.
    W[k] is an integer where bit m is 1 if wire k is 1 in input m.
    """
    total = 1 << N
    W = [0] * N
    # We can construct W[k] by setting bits.
    # Bit m of W[k] is 1 if (m >> k) & 1 is 1.
    # This is equivalent to a pattern of 0s and 1s.
    # For N=17, we can just loop or use bit tricks.
    # Looping 131k times is fine.
    for m in range(total):
        for k in range(N):
            if (m >> k) & 1:
                W[k] |= (1 << m)
    return W

def _verify_fast(comps):
    """
    Verify sorting network using bit-parallel simulation.
    """
    if not comps:
        return N <= 1
    
    W = _init_bitsets()
    
    for i, j in comps:
        # min goes to i, max goes to j
        # min(a, b) = a & b
        # max(a, b) = a | b
        wi = W[i]
        wj = W[j]
        W[i] = wi & wj
        W[j] = wi | wj
        
    # Check if all outputs are sorted
    # For each k, Out[k] should be <= Out[k+1]
    # i.e., (Out[k] & ~Out[k+1]) == 0
    for k in range(N - 1):
        if (W[k] & ~W[k+1]) != 0:
            return False
    return True

def _build_seed_network():
    """
    Construct a Bitonic Sort network for N=17.
    Bitonic sort is O(N^2 log N) comparators? No, O(N log^2 N).
    For N=17, it should be around 100-120 comparators.
    """
    comps = []
    
    def bitonic_sort(n, offset, direction):
        """
        Sort n elements starting at offset.
        direction: 1 for ascending, 0 for descending.
        """
        if n <= 1:
            return
        
        k = 1
        while k < n:
            # Merge phase
            # For i in 0..n/2-1
            # Compare (offset + i, offset + i + n/2)
            # Direction depends on current phase and global direction
            # Standard Bitonic Merge:
            # For i in 0..n/2-1:
            #   if (i // (k/2)) % 2 == 0: compare (offset+i, offset+i+n/2) with dir
            #   else: compare (offset+i, offset+i+n/2) with !dir
            # Wait, the standard bitonic merge is:
            # for i in range(0, n//2):
            #    if (i // (k//2)) % 2 == 0:
            #        a = offset + i
            #        b = offset + i + n//2
            #        if direction == 1:
            #            comps.append((a, b))
            #        else:
            #            comps.append((b, a))
            #    else:
            #        a = offset + i
            #        b = offset + i + n//2
            #        if direction == 0:
            #            comps.append((a, b))
            #        else:
            #            comps.append((b, a))
            
            half = n // 2
            for i in range(half):
                a = offset + i
                b = offset + i + half
                # Determine local direction
                # In bitonic merge, the direction of comparison alternates based on i and k
                # Specifically, for the merge step of size 2k:
                # The sub-sequences are already sorted.
                # We compare (i, i+k) for i in 0..k-1? No, i in 0..n/2-1.
                # The pattern is:
                # For j in 0..n/2-1:
                #   if (j // (k/2)) % 2 == 0:
                #      sort (j, j+k) in ascending order (if global is asc)
                #   else:
                #      sort (j, j+k) in descending order (if global is asc)
                
                # Let's use a simpler recursive definition if needed, 
                # but iterative is faster.
                # Actually, let's use the standard recursive bitonic sort which is easier to get right.
                pass
        
        # Let's stick to a known correct implementation.
        # I will implement a standard Bitonic Sort network generator.
        
        pass

    # Let's use Odd-Merge as it's often smaller for small N or similar.
    # Or just use a known good network for 17.
    # Actually, Batcher's Bitonic Sort for N=16 is 64 comparators.
    # For N=17, we can pad to 18? No, N must be 17.
    # We can sort 16 and merge the 17th?
    # Or use Odd-Merge.
    
    # Let's implement Odd-Merge properly.
    
    def odd_merge(n, offset):
        if n <= 1:
            return
        if n == 2:
            a, b = offset, offset + 1
            comps.append((a, b))
            return
            
        n_odd = (n + 1) // 2
        n_even = n // 2
        
        # Odd indices: 0, 2, 4...
        odd_wires = [offset + 2*i for i in range(n_odd)]
        # Even indices: 1, 3, 5...
        even_wires = [offset + 2*i + 1 for i in range(n_even)]
        
        # Recursively sort odd and even subsequences
        # We need to map the recursive calls back to the actual wire indices.
        # The recursive call sorts a virtual array of length n_odd or n_even.
        # We can just call a helper that takes a list of wires.
        
        def sort_wires(wires, length):
            if length <= 1:
                return
            if length == 2:
                a, b = wires[0], wires[1]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
                return
                
            half_odd = (length + 1) // 2
            half_even = length // 2
            
            sub_odd = [wires[2*i] for i in range(half_odd)]
            sub_even = [wires[2*i+1] for i in range(half_even)]
            
            sort_wires(sub_odd, half_odd)
            sort_wires(sub_even, half_even)
            
            # Merge
            m = min(half_odd, half_even)
            for i in range(m):
                a = sub_odd[i]
                b = sub_even[i]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
            for i in range(m - 1):
                a = sub_odd[i+1]
                b = sub_even[i]
                if a < b:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
                    
        sort_wires(list(range(N)), N)
        
    odd_merge(N, 0)
    return comps

def _local_search(comps, deadline):
    """
    Local search with fast verification.
    """
    comps = list(comps)
    best = list(comps)
    best_len = len(comps)
    
    # Pre-calculate all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        moved = False
        n = len(comps)
        
        # Phase 1: Removals
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return best
            cand = comps[:idx] + comps[idx+1:]
            if _verify_fast(cand):
                comps = cand
                best = list(comps)
                best_len = len(comps)
                moved = True
                break
                
        if moved:
            continue
            
        # Phase 2: Swaps
        # Try swapping two comparators
        # To save time, only try a subset or random swaps if n is large
        # n is around 100, so n^2 is 10000. Verify is fast.
        for i in range(n):
            if time.time() > deadline:
                return best
            for j in range(i+1, n):
                if time.time() > deadline:
                    return best
                cand = comps[:]
                cand[i], cand[j] = cand[j], cand[i]
                if _verify_fast(cand):
                    # Try to remove after swap
                    # Simple greedy removal on cand
                    reduced = list(cand)
                    # One pass of removal
                    for k in range(len(reduced) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        temp = reduced[:k] + reduced[k+1:]
                        if _verify_fast(temp):
                            reduced = temp
                    if len(reduced) < best_len:
                        comps = reduced
                        best = list(comps)
                        best_len = len(comps)
                        moved = True
                        break
                if time.time() > deadline:
                    return best
            if moved:
                break
                
        if moved:
            continue
            
        # Phase 3: 2-for-1 moves (Add 1, Remove 2)
        # This is harder. 
        # Alternative: Random restarts or random local moves.
        # Let's try inserting a random comparator and then removing greedily.
        # If we can add 1 and remove 2, we net -1.
        
        # Try adding a random comparator
        # Then try to remove 2 comparators.
        
        # Simpler: Just try random swaps and removals.
        # If stuck, try a random insertion that might enable removals.
        
        # Let's do a random walk if stuck.
        # Pick a random comparator to add.
        # Then run greedy removal.
        # If result is better, accept.
        
        for _ in range(10): # 10 random attempts
            if time.time() > deadline:
                return best
            rand_comp = random.choice(all_comps)
            if rand_comp in comps:
                continue
            # Insert at random position
            pos = random.randint(0, len(comps))
            cand = comps[:pos] + [rand_comp] + comps[pos:]
            
            # Greedy removal
            improved = True
            while improved:
                improved = False
                for k in range(len(cand) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    temp = cand[:k] + cand[k+1:]
                    if _verify_fast(temp):
                        cand = temp
                        improved = True
                        break
                if time.time() > deadline:
                    break
            
            if len(cand) < best_len:
                comps = cand
                best = list(comps)
                best_len = len(comps)
                moved = True
                break
                
        if not moved:
            # If we are stuck, maybe try a different seed or break.
            # For now, break to avoid infinite loop if no improvement found.
            # But we should keep trying random moves if time allows.
            # Let's just break for now, as the main loop will restart if we restructure.
            # Actually, let's just break to return best.
            break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build seed
    comps = _build_seed_network()
    
    if not _verify_fast(comps):
        # Fallback: Odd-Even Transposition
        comps = [(i, i+1) for phase in range(N) for i in range(phase % 2, N-1, 2)]
        
    # Local search
    comps = _local_search(comps, deadline)
    
    return comps
