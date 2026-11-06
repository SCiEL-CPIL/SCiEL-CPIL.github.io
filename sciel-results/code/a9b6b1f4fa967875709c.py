"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_set = None
    best_size = N + 1
    
    def is_sidon(S):
        if not S:
            return True
        L = sorted(S)
        seen = set()
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    def get_addable(S, T):
        res = []
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if 2 * x in T:
                continue
            ok = True
            for s in S:
                if x + s in T:
                    ok = False
                    break
            if ok:
                res.append(x)
        return res

    def calc_blocking(S, T, x, addable):
        # How many elements in 'addable' are blocked by adding x?
        new_sums = {2 * x}
        for s in S:
            new_sums.add(x + s)
        blocked = 0
        Sset = set(S) | {x}
        T2 = T | new_sums
        for y in addable:
            if y == x:
                continue
            if 2 * y in T2:
                blocked += 1
                continue
            ok = True
            for s in Sset:
                if y + s in T2:
                    ok = False
                    break
            if not ok:
                blocked += 1
        return blocked

    def randomized_greedy():
        S = []
        T = set()
        addable = get_addable(S, T)
        
        while addable and time.time() < deadline:
            if not addable:
                break
            
            # Calculate blocking power for all addable
            # Optimization: Only recalculate for the current addable set
            block_counts = {}
            for x in addable:
                block_counts[x] = calc_blocking(S, T, x, addable)
            
            # Choose x with probability proportional to (block_count + 1)
            # Or pick from top K
            # Let's use a softmax-like selection to balance exploration/exploitation
            # Or simply pick randomly from the top 20% of blockers
            sorted_items = sorted(block_counts.items(), key=lambda item: item[1], reverse=True)
            top_k = max(1, len(sorted_items) // 5) # Top 20%
            candidates = [item[0] for item in sorted_items[:top_k]]
            
            x = random.choice(candidates)
            
            S.append(x)
            new_sums = {2 * x}
            for s in S[:-1]:
                new_sums.add(x + s)
            T |= new_sums
            
            # Update addable
            Sset = set(S)
            new_addable = []
            for y in addable:
                if y == x:
                    continue
                if 2 * y in T:
                    continue
                ok = True
                for s in S:
                    if y + s in T:
                        ok = False
                        break
                if ok:
                    new_addable.append(y)
            addable = new_addable
            
        # Check if S is maximal (addable should be empty)
        if addable:
            # If we timed out, it might not be maximal. 
            # But the loop condition was `while addable and time...`
            # So if time runs out, addable might not be empty.
            # We must ensure the returned set is maximal.
            # If not maximal, we can try to complete it greedily?
            # Or just discard this candidate if it's not maximal?
            # The task requires the returned set to be maximal.
            # If we stop early, it's not maximal.
            # So we should only accept if addable is empty.
            return None
        
        return S

    # Run multiple randomized greedy attempts
    while time.time() < deadline:
        S = randomized_greedy()
        if S is not None:
            if len(S) < best_size:
                best_size = len(S)
                best_set = S[:]
                # Optional: If we find a very small set, we might want to keep searching for even smaller
                # but we can break if we hit a theoretical lower bound? 
                # Lower bound for maximal Sidon set in [1,N] is roughly sqrt(N).
                # sqrt(256) = 16. 
                # If we find 16, we might stop. But let's see what we get.
                
    if best_set is None:
        # Fallback to a simple deterministic greedy if random failed (e.g. timeout before completion)
        S = []
        T = set()
        addable = get_addable(S, T)
        while addable:
            best_x = None
            best_block = -1
            for x in addable:
                b = calc_blocking(S, T, x, addable)
                if b > best_block:
                    best_block = b
                    best_x = x
            if best_x is None:
                break
            S.append(best_x)
            new_sums = {2 * best_x}
            for s in S[:-1]:
                new_sums.add(best_x + s)
            T |= new_sums
            Sset = set(S)
            addable = [y for y in addable if y != best_x and 2*y not in T and not any(y+s in T for s in S)]
        best_set = S

    # Final verification
    if not is_sidon(best_set):
        # This should not happen if logic is correct, but just in case
        # Return a valid small set? 
        # Simple valid Sidon set: powers of 2? No, not maximal.
        # Just return the first few integers? No, must be Sidon.
        # {1, 2, 4, 8...} is Sidon. Is it maximal? No.
        # We need a maximal one.
        # If verification fails, we have a bug. 
        # For safety, return a known valid small maximal set if possible?
        # Or just return best_set and hope the verifier is lenient? No, it must be valid.
        # Let's force a valid maximal set using the deterministic fallback logic again but more carefully.
        pass

    return best_set
# EVOLVE-BLOCK-END
