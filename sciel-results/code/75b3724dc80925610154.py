"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds (Safe margin for 60s/90s limits)


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])
    if diffsA & diffsB:
        return False
    # Check all elements in [1, N]
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
    return True


def _algebraic_init(rng):
    """
    Structured algebraic construction using Z_16 x Z_16.
    Map [1,256] to Z_16 x Z_16 via x -> (x-1) // 16, (x-1) % 16.
    Define A as elements where first coord is in {0..7} (a subgroup-like set)
    and B where first coord is in {8..15}.
    Then apply a random bijection-like permutation to break structure while
    preserving approximate difference disjointness.
    """
    # Build a random permutation of [1, 256]
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    
    # Try several structured patterns
    best_A, best_B = [], []
    best_size = 0
    
    for pattern_idx in range(8):
        # Different algebraic structures
        if pattern_idx == 0:
            # A: first 8 rows, B: last 8 rows
            A_idx = [i for i in range(256) if (i // 16) < 8]
            B_idx = [i for i in range(256) if (i // 16) >= 8]
        elif pattern_idx == 1:
            # A: even rows, B: odd rows
            A_idx = [i for i in range(256) if (i // 16) % 2 == 0]
            B_idx = [i for i in range(256) if (i // 16) % 2 == 1]
        elif pattern_idx == 2:
            # A: first 4 cols of each row, B: last 12 cols
            A_idx = [i for i in range(256) if (i % 16) < 4]
            B_idx = [i for i in range(256) if (i % 16) >= 4]
        elif pattern_idx == 3:
            # A: rows 0-3, B: rows 4-11
            A_idx = [i for i in range(256) if (i // 16) < 4]
            B_idx = [i for i in range(256) if 4 <= (i // 16) < 12]
        elif pattern_idx == 4:
            # A: diagonal band, B: anti-diagonal band
            A_idx = [i for i in range(256) if abs((i // 16) - (i % 16)) <= 3]
            B_idx = [i for i in range(256) if abs((i // 16) + (i % 16) - 15) <= 3]
        elif pattern_idx == 5:
            # A: rows 0-5, B: rows 10-15
            A_idx = [i for i in range(256) if (i // 16) < 6]
            B_idx = [i for i in range(256) if (i // 16) >= 10]
        elif pattern_idx == 6:
            # A: cols 0-7, B: cols 8-15
            A_idx = [i for i in range(256) if (i % 16) < 8]
            B_idx = [i for i in range(256) if (i % 16) >= 8]
        else:
            # A: rows 0-7 with cols 0-7, B: rows 8-15 with cols 8-15
            A_idx = [i for i in range(256) if (i // 16) < 8 and (i % 16) < 8]
            B_idx = [i for i in range(256) if (i // 16) >= 8 and (i % 16) >= 8]
        
        # Map indices to actual values via permutation
        A = [perm[i] for i in A_idx]
        B = [perm[i] for i in B_idx]
        
        # Validate and keep if valid
        if _validate(A, B):
            sz = len(A) + len(B)
            if sz > best_size:
                best_size = sz
                best_A, best_B = list(A), list(B)
    
    return best_A, best_B


def _score_candidate(x, A, B, sumsA, sumsB, diffsA, diffsB):
    """
    Score a candidate x for addition: lower is better.
    Penalizes conflicts and rewards flexibility.
    """
    score = 0
    # Count how many sums would conflict
    if len(A) < MAX_EACH:
        conflict_A = 0
        if (x + x) in sumsA:
            conflict_A += 10
        else:
            for a in A:
                if (x + a) in sumsA:
                    conflict_A += 1
        # Count new diffs that conflict with B
        new_diffs_conflict = 0
        for a in A:
            d = abs(a - x)
            if d > 0 and (diffsB >> d) & 1:
                new_diffs_conflict += 1
        score_A = conflict_A * 2 + new_diffs_conflict
    else:
        score_A = 1000
    
    if len(B) < MAX_EACH:
        conflict_B = 0
        if (x + x) in sumsB:
            conflict_B += 10
        else:
            for b in B:
                if (x + b) in sumsB:
                    conflict_B += 1
        new_diffs_conflict = 0
        for b in B:
            d = abs(b - x)
            if d > 0 and (diffsA >> d) & 1:
                new_diffs_conflict += 1
        score_B = conflict_B * 2 + new_diffs_conflict
    else:
        score_B = 1000
    
    return min(score_A, score_B)


def _fast_greedy(t0, rng, seed_offset=0):
    """
    Fast greedy builder using incremental sum sets and bitmask differences.
    Uses smart candidate ordering.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = 0  # Bitmask for differences in A
    diffsB = 0  # Bitmask for differences in B

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)

    # To add variation, we can bias the preference
    prefer_A = rng.random() < 0.5

    # We'll process in rounds, scoring candidates each round
    remaining = list(candidates)
    
    while remaining:
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Score all remaining candidates
        scored = []
        for x in remaining:
            sc = _score_candidate(x, A, B, sumsA, sumsB, diffsA, diffsB)
            scored.append((sc, x))
        
        # Sort by score (lower is better), with random tie-breaking
        rng.shuffle(scored)
        scored.sort(key=lambda t: t[0])
        
        # Try to add the best candidate
        added = False
        for sc, x in scored:
            if sc >= 1000:
                break  # No valid side available
            
            # Determine which side to try based on score
            if prefer_A:
                sides = [0, 1]
            else:
                sides = [1, 0]

            for side in sides:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    # Check Sidon condition for A with x
                    valid_sidon_A = True
                    if (x + x) in sumsA:
                        valid_sidon_A = False
                    else:
                        for a in A:
                            if (x + a) in sumsA:
                                valid_sidon_A = False
                                break

                    if valid_sidon_A:
                        # Check difference constraints
                        new_diffs_mask = 0
                        for a in A:
                            d = abs(a - x)
                            if d > 0:
                                new_diffs_mask |= (1 << d)

                        if not (new_diffs_mask & diffsB):
                            # Valid! Add x to A
                            A.append(x)
                            sumsA.add(x + x)
                            for a in A[:-1]:
                                sumsA.add(x + a)
                            diffsA |= new_diffs_mask
                            added = True
                            break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    # Check Sidon condition for B with x
                    valid_sidon_B = True
                    if (x + x) in sumsB:
                        valid_sidon_B = False
                    else:
                        for b in B:
                            if (x + b) in sumsB:
                                valid_sidon_B = False
                                break

                    if valid_sidon_B:
                        new_diffs_mask = 0
                        for b in B:
                            d = abs(b - x)
                            if d > 0:
                                new_diffs_mask |= (1 << d)

                        if not (new_diffs_mask & diffsA):
                            # Valid! Add x to B
                            B.append(x)
                            sumsB.add(x + x)
                            for b in B[:-1]:
                                sumsB.add(x + b)
                            diffsB |= new_diffs_mask
                            added = True
                            break
            
            if added:
                break
        
        # Remove added candidate from remaining
        if added:
            # Find which x was added
            if len(A) > 0 and A[-1] in remaining:
                remaining.remove(A[-1])
            elif len(B) > 0 and B[-1] in remaining:
                remaining.remove(B[-1])
        else:
            break  # No more additions possible

    return A, B


def _local_improve(A, B, t0):
    """
    Try to improve A and B by adding new ones and swapping elements.
    Uses bitmask diffs and incremental sums for speed.
    """
    if not A or not B:
        return A, B

    # Recompute state
    sumsA = set()
    sumsB = set()
    diffsA = 0
    diffsB = 0

    for x in A:
        for y in A:
            if x <= y:
                sumsA.add(x + y)
            if x != y:
                d = abs(x - y)
                diffsA |= (1 << d)
    for x in B:
        for y in B:
            if x <= y:
                sumsB.add(x + y)
            if x != y:
                d = abs(x - y)
                diffsB |= (1 << d)

    used = set(A) | set(B)

    # Phase 1: Try to add more elements
    for x in range(1, N + 1):
        if time.time() - t0 > DEADLINE - 2:
            break
        if x in used:
            continue

        # Try A
        if len(A) < MAX_EACH:
            valid = True
            if (x + x) in sumsA:
                valid = False
            else:
                for a in A:
                    if (x + a) in sumsA:
                        valid = False
                        break
            if valid:
                new_diffs_mask = 0
                for a in A:
                    d = abs(a - x)
                    if d > 0:
                        new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & diffsB):
                    A.append(x)
                    used.add(x)
                    sumsA.add(x + x)
                    for a in A[:-1]:
                        sumsA.add(x + a)
                    diffsA |= new_diffs_mask
                    continue

        # Try B
        if len(B) < MAX_EACH:
            valid = True
            if (x + x) in sumsB:
                valid = False
            else:
                for b in B:
                    if (x + b) in sumsB:
                        valid = False
                        break
            if valid:
                new_diffs_mask = 0
                for b in B:
                    d = abs(b - x)
                    if d > 0:
                        new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & diffsA):
                    B.append(x)
                    used.add(x)
                    sumsB.add(x + x)
                    for b in B[:-1]:
                        sumsB.add(x + b)
                    diffsB |= new_diffs_mask
                    continue

    # Phase 2: Swap-based local search
    # Try to swap elements between A and B, or move elements to free up space
    for _iter in range(20):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        improved = False
        
        # Try moving an element from A to B or vice versa
        for i in range(len(A)):
            if time.time() - t0 > DEADLINE - 3:
                break
            x = A[i]
            # Check if x can be moved to B
            if len(B) < MAX_EACH:
                # Check if removing x from A keeps A valid (it will)
                # Check if adding x to B is valid
                valid_B = True
                if (x + x) in sumsB:
                    valid_B = False
                else:
                    for b in B:
                        if (x + b) in sumsB:
                            valid_B = False
                            break
                if valid_B:
                    new_diffs_mask = 0
                    for b in B:
                        d = abs(b - x)
                        if d > 0:
                            new_diffs_mask |= (1 << d)
                    if not (new_diffs_mask & diffsA):
                        # Move x from A to B
                        A.pop(i)
                        B.append(x)
                        # Recompute sumsA
                        sumsA = set()
                        for a1 in A:
                            for a2 in A:
                                if a1 <= a2:
                                    sumsA.add(a1 + a2)
                        # Add x to B's sums
                        sumsB.add(x + x)
                        for b in B[:-1]:
                            sumsB.add(x + b)
                        # Update diffs
                        diffsA = 0
                        for a1 in A:
                            for a2 in A:
                                if a1 != a2:
                                    d = abs(a1 - a2)
                                    diffsA |= (1 << d)
                        diffsB |= new_diffs_mask
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            for i in range(len(B)):
                if time.time() - t0 > DEADLINE - 3:
                    break
                x = B[i]
                # Check if x can be moved to A
                if len(A) < MAX_EACH:
                    valid_A = True
                    if (x + x) in sumsA:
                        valid_A = False
                    else:
                        for a in A:
                            if (x + a) in sumsA:
                                valid_A = False
                                break
                    if valid_A:
                        new_diffs_mask = 0
                        for a in A:
                            d = abs(a - x)
                            if d > 0:
                                new_diffs_mask |= (1 << d)
                        if not (new_diffs_mask & diffsB):
                            # Move x from B to A
                            B.pop(i)
                            A.append(x)
                            # Recompute sumsB
                            sumsB = set()
                            for b1 in B:
                                for b2 in B:
                                    if b1 <= b2:
                                        sumsB.add(b1 + b2)
                            # Add x to A's sums
                            sumsA.add(x + x)
                            for a in A[:-1]:
                                sumsA.add(x + a)
                            # Update diffs
                            diffsB = 0
                            for b1 in B:
                                for b2 in B:
                                    if b1 != b2:
                                        d = abs(b1 - b2)
                                        diffsB |= (1 << d)
                            diffsA |= new_diffs_mask
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break

    return A, B


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # First, try algebraic initialization
    alg_A, alg_B = _algebraic_init(rng)
    if alg_A and alg_B:
        alg_A, alg_B = _local_improve(alg_A, alg_B, t0)
        if _validate(alg_A, alg_B):
            best_size = len(alg_A) + len(alg_B)
            best_A, best_B = list(alg_A), list(alg_B)
            if best_size == 128:
                return [best_A, best_B]

    # Run multiple fast greedy restarts
    i = 0
    while time.time() - t0 < DEADLINE - 2:
        # Use a different seed for each restart to vary the shuffle
        r = random.Random(42 + i)
        A, B = _fast_greedy(t0, r)
        
        if not A or not B:
            i += 1
            continue
        
        # Local improvement
        A, B = _local_improve(A, B, t0)
        
        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)
            
            # If we hit the max possible, we can stop early? Max is 128.
            if best_size == 128:
                break
        
        i += 1

    # Ensure nonempty and valid
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    if not _validate(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
