"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill
    
    def check_sidon(S):
        """Check if S is a valid Sidon set."""
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def can_add(S, sums, x):
        """Check if x can be added to S (with its sums) maintaining Sidon property."""
        for a in S:
            s = x + a
            if s in sums:
                return False
        s = 2 * x
        if s in sums:
            return False
        return True
    
    def count_blocked(S, sums, x):
        """Count how many elements from [1,N] would be blocked if x is added to S."""
        if not can_add(S, sums, x):
            return -1
        new_sums = sums | {x + a for a in S} | {2 * x}
        new_S = S + [x]
        blocked = 0
        for y in range(1, N + 1):
            if y in S or y == x:
                continue
            # Check if y can be added to new_S
            ok = True
            for a in new_S:
                if x + y in new_sums and a != x:
                    # x+y is in new_sums, but we need to check if it conflicts
                    pass
                s = y + a
                if s in new_sums:
                    ok = False
                    break
            if ok:
                if 2 * y in new_sums:
                    ok = False
            if not ok:
                blocked += 1
        return blocked
    
    # Greedy: build a small maximal Sidon set
    # At each step, pick the element that blocks the most remaining elements
    S = []
    sums = set()
    
    while True:
        if time.time() > deadline:
            break
        best_x = -1
        best_blocked = -1
        for x in range(1, N + 1):
            if x in S:
                continue
            b = count_blocked(S, sums, x)
            if b > best_blocked:
                best_blocked = b
                best_x = x
        if best_x == -1:
            break
        # Add best_x
        S.append(best_x)
        for a in S:
            sums.add(best_x + a)
        sums.add(2 * best_x)
    
    # Verify it's maximal
    # Try adding each missing element
    for x in range(1, N + 1):
        if x not in S:
            if can_add(S, sums, x):
                # Not maximal - this shouldn't happen if we ran to completion
                # But just in case, we could try to fix it
                pass
    
    return S
# EVOLVE-BLOCK-END
