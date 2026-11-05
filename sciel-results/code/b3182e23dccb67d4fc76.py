"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 132  # safety margin before 150s kill

    base = 51975

    # All odd divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # INCREASING order: small moduli claim residues first

    covered = np.zeros(base, dtype=np.uint8)
    classes = []

    # ---- Phase 1: Greedy, increasing modulus, vectorized residue scoring ----
    for m in divisors:
        if time.time() > deadline - 20:
            break
        # For each residue r, count uncovered positions r, r+m, r+2m, ...
        # positions for residue r is covered[r::m]; uncovered = len - sum
        # Vectorize over all residues: build a matrix of counts.
        if m <= 400:
            # m small: build array of counts for all residues at once
            # For residue r, the covered positions are r, r+m, ...
            # We can compute per-residue sums via reshaping trick is complex;
            # instead loop but vectorize the sum (fast in numpy).
            best_r = 0
            best_count = -1
            for r in range(m):
                seg = covered[r::m]
                cnt = m and (len(seg) - int(seg.sum()))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
        else:
            # m large: sample residues (few positions each)
            best_r = 0
            best_count = -1
            # deterministic candidate set
            cands = set()
            for k in range(0, min(m, 64)):
                cands.add(k)
            for k in range(1, min(m, 32)):
                cands.add((k * m) // 32)
            rng = np.random.RandomState(len(classes) * 104729 + 7)
            for _ in range(40):
                cands.add(int(rng.randint(m)))
            for r in cands:
                seg = covered[r::m]
                cnt = len(seg) - int(seg.sum())
                if cnt > best_count:
                    best_count = cnt
                    best_r = r

        classes.append([m, best_r])
        covered[best_r::m] = 1

    def recompute_coverage(cls):
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in cls:
            cov[r::m] = 1
        return cov

    # ---- Phase 2: Local search (vectorized, recompute once per pass) ----
    if time.time() < deadline - 12:
        max_passes = 6
        for p in range(max_passes):
            if time.time() > deadline - 12:
                break
            improved = False
            cov = recompute_coverage(classes)
            for i in range(len(classes)):
                if time.time() > deadline - 12:
                    break
                m, r = classes[i]
                # coverage without this class
                cov_without = np.zeros(base, dtype=np.uint8)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        cov_without[r2::m2] = 1
                base_uncov = int((cov_without == 0).sum())
                best_r = r
                best_uncov = base_uncov
                # For each candidate residue, new_uncovered = base_uncov - (newly covered)
                # newly covered = positions in r'::m that are 0 in cov_without
                if m <= 250:
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        newly = int((cov_without[r2::m] == 0).sum())
                        unc = base_uncov - newly
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2
                else:
                    cands = set()
                    for k in range(0, min(m, 48)):
                        cands.add(k)
                    for k in range(1, min(m, 24)):
                        cands.add((k * m) // 24)
                    rng = np.random.RandomState(i * 98765 + p * 31 + 4321)
                    for _ in range(40):
                        cands.add(int(rng.randint(m)))
                    for r2 in cands:
                        if r2 == r:
                            continue
                        newly = int((cov_without[r2::m] == 0).sum())
                        unc = base_uncov - newly
                        if unc < best_uncov:
                            best_uncov = unc
                            best_r = r2
                if best_r != r:
                    classes[i][1] = best_r
                    improved = True
            if not improved:
                break

    # ---- Phase 3: Remove redundant classes ----
    if time.time() < deadline - 8:
        removed = True
        while removed and time.time() < deadline - 8:
            removed = False
            cov = recompute_coverage(classes)
            cur_uncov = int((cov == 0).sum())
            for i in range(len(classes) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break
                cov_without = np.zeros(base, dtype=np.uint8)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        cov_without[r2::m2] = 1
                unc_without = int((cov_without == 0).sum())
                if unc_without <= cur_uncov:
                    classes.pop(i)
                    removed = True
                    cur_uncov = unc_without

    # ---- Phase 4: Final high-precision polish over small moduli (exact) ----
    if time.time() < deadline - 4:
        # exact sweep: for each small modulus, recompute best residue exactly
        for i in range(len(classes)):
            if time.time() > deadline - 4:
                break
            m, r = classes[i]
            if m > 300:
                continue
            cov_without = np.zeros(base, dtype=np.uint8)
            for j, (m2, r2) in enumerate(classes):
                if j != i:
                    cov_without[r2::m2] = 1
            base_uncov = int((cov_without == 0).sum())
            best_r = r
            best_uncov = base_uncov
            for r2 in range(m):
                newly = int((cov_without[r2::m] == 0).sum())
                unc = base_uncov - newly
                if unc < best_uncov:
                    best_uncov = unc
                    best_r = r2
            if best_r != r:
                classes[i][1] = best_r

    return classes
# EVOLVE-BLOCK-END
