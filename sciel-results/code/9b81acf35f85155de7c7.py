"""Combined program: upper-tail seed + fast residue-mask conflict checks + local-search polish."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    n = 256

    # ------------------------------------------------------------------
    # Precompute conflict masks:
    #   conflict[a][b] = bitmask of c-values (1..n) such that a | (b + c).
    #   Used for: "if a and b are in S, then no c in conflict[a][b] may be in S"
    #   and symmetric queries when adding a new element.
    # ------------------------------------------------------------------
    conflict = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            s = b  # b + c, c ranges 1..n
            # c such that (b+c) % a == 0  =>  c ≡ (-b) mod a,  1 <= c <= n
            r = (-b) % a
            # c values: r, r+a, r+2a, ... but c must be >= 1
            # if r == 0, first valid c is a; else first is r
            first = r if r != 0 else a
            mask = 0
            for c in range(first, n + 1, a):
                mask |= (1 << c)
            conflict[a][b] = mask

    # ------------------------------------------------------------------
    # Fast validity check for a set S (bitmask representation)
    # ------------------------------------------------------------------
    def is_valid_mask(mask):
        """Check if bitmask mask (bits 1..n set) is a valid set."""
        S = [i for i in range(1, n + 1) if (mask >> i) & 1]
        m = len(S)
        for i in range(m):
            a = S[i]
            for j in range(i + 1, m):
                b = S[j]
                # check b+b: c=b
                if (mask >> b) & 1 and ((2 * b) % a == 0):
                    # b+b conflict: a | (b+b)
                    return False
                # check b+c for c > b
                cm = conflict[a][b] & mask
                # remove bits <= b (we only care about c >= b, but b+b already checked)
                # Actually conflict[a][b] includes all c with a|(b+c), we need c >= b
                # Let's just check: is there any c in S with c >= b and c in conflict[a][b]?
                # mask has bits for S. conflict[a][b] has bits for valid c.
                # Intersection gives c in S that conflict. But we need c >= b.
                # Since b is in S and b+b was checked, we can check c > b.
                # Simpler: check all c in S with c >= b.
                # Use bit trick: mask_c = mask & conflict[a][b] & (~((1 << b) - 1))
                # This keeps bits >= b.
                if cm & (~((1 << b) - 1)) & mask:
                    # There exists c >= b in S with a | (b+c)
                    # But wait, b itself: if b in conflict[a][b], that means a|(b+b), already checked.
                    # Let's just return False if any such c exists.
                    return False
        return True

    # ------------------------------------------------------------------
    # Fast can_add: check if adding x to mask is safe
    # ------------------------------------------------------------------
    def can_add(mask, x):
        """Check if adding element x (not in mask) keeps the set valid."""
        if (mask >> x) & 1:
            return False

        # Case 1: x is 'a' (smallest). Need: for all b,c in S with x < b <= c, x ∤ (b+c).
        # Equivalent: for all b in S with b > x, no c in S with c >= b and x | (b+c).
        # Use conflict[x][b] & mask, check if any c >= b is set.
        for b in range(x + 1, n + 1):
            if (mask >> b) & 1:
                # Check if any c in S with c >= b and c in conflict[x][b]
                if (conflict[x][b] & mask) & (~((1 << b) - 1)):
                    return False

        # Case 2: x is 'b' (middle). Need: for all a in S with a < x, and c in S with c >= x,
        # a ∤ (x + c).
        # Equivalent: for all a in S with a < x, no c in S with c >= x and a | (x+c).
        # conflict[a][x] gives c-values where a | (x+c). Check intersection with mask & (bits >= x).
        ge_x_mask = ~((1 << x) - 1)  # bits >= x
        for a in range(1, x):
            if (mask >> a) & 1:
                if (conflict[a][x] & mask) & ge_x_mask:
                    return False

        # Case 3: x is 'c' (largest). Need: for all a < b in S with b <= x, a ∤ (b + x).
        # Equivalent: for all a in S, for all b in S with a < b <= x, a ∤ (b+x).
        # conflict[a][b] gives c-values where a | (b+c). We need x in conflict[a][b].
        for a in range(1, x):
            if (mask >> a) & 1:
                for b in range(a + 1, x + 1):
                    if (mask >> b) & 1:
                        if (conflict[a][b] >> x) & 1:
                            return False

        return True

    # ------------------------------------------------------------------
    # Local search: try add, remove, swap to improve size
    # ------------------------------------------------------------------
    def local_search(mask, max_iters=2000):
        """Try to improve mask via add/remove/swap moves."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break

            # Try adding elements
            candidates = [x for x in range(1, n + 1) if not ((mask >> x) & 1)]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(mask, x):
                    mask |= (1 << x)
                    improved = True

            # Try removing one element to allow adding two
            S = [x for x in range(1, n + 1) if (mask >> x) & 1]
            if len(S) > 2:
                for x in S:
                    if time.time() > deadline:
                        break
                    trial = mask & ~(1 << x)
                    # Try to add two elements to trial
                    added = 0
                    candidates2 = [y for y in range(1, n + 1) if not ((trial >> y) & 1)]
                    random.shuffle(candidates2)
                    for y in candidates2:
                        if added >= 2:
                            break
                        if can_add(trial, y):
                            trial |= (1 << y)
                            added += 1
                    if added >= 2:
                        mask = trial
                        improved = True
                        break

        return mask

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    best_mask = 0
    best_size = 0

    # Strategy 1: Upper-tail seeds with greedy expansion + local search
    for restart in range(60):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 137 + 42)

        # Start with upper tail
        tail_start = rng.randint(150, 200)
        mask = 0
        for x in range(tail_start, n + 1):
            mask |= (1 << x)

        # Verify and fix
        if not is_valid_mask(mask):
            # Remove problematic elements
            S = sorted([x for x in range(1, n + 1) if (mask >> x) & 1])
            new_mask = 0
            for x in S:
                if can_add(new_mask, x):
                    new_mask |= (1 << x)
            mask = new_mask

        # Greedy add from below
        candidates = [x for x in range(1, tail_start) if not ((mask >> x) & 1)]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(mask, x):
                mask |= (1 << x)

        # Second pass: try all remaining elements
        candidates2 = [x for x in range(1, n + 1) if not ((mask >> x) & 1)]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add(mask, x):
                mask |= (1 << x)

        # Local search polish
        mask = local_search(mask, max_iters=1000)

        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    # Strategy 2: Random subset seeds
    for restart in range(40):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 211 + 99)

        # Start with random subset of size 30-60
        k = rng.randint(30, 60)
        S = rng.sample(range(1, n + 1), k)
        mask = 0
        for x in sorted(S):
            if can_add(mask, x):
                mask |= (1 << x)

        # Greedy add
        candidates = [x for x in range(1, n + 1) if not ((mask >> x) & 1)]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(mask, x):
                mask |= (1 << x)

        # Local search
        mask = local_search(mask, max_iters=1000)

        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    # Final polish on best
    if time.time() < deadline - 5.0:
        best_mask = local_search(best_mask, max_iters=3000)
        best_size = bin(best_mask).count('1')

    # Convert to list
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    if not result:
        result = list(range(192, 257))

    return result
# EVOLVE-BLOCK-END
