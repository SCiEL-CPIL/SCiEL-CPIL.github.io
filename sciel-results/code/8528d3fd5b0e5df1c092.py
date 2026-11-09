import time
import random
import numpy as np

N = 51975
# All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, greater than 1
_divs = [3**a * 5**b * 7**c * 11**d for a in range(4) for b in range(3) for c in range(2) for d in range(2) if 3**a * 5**b * 7**c * 11**d > 1]
DIVISORS = sorted(_divs)

# Precompute index lists for each modulus
INDICES = {}
for m in DIVISORS:
    INDICES[m] = list(range(N // m))  # length = N // m

def _score(classes):
    """Return number of uncovered residues."""
    covered = np.zeros(N, dtype=np.uint8)
    for m, r in classes:
        idx = r + m * np.arange(N // m)
        covered[idx] = 1
    return int(N - covered.sum())

def _greedy():
    """Greedy construction: for each modulus in increasing order, pick residue covering most uncovered."""
    covered = np.zeros(N, dtype=np.uint8)
    classes = []
    for m in DIVISORS:
        best_r = 0
        best_count = -1
        cnt_arr = np.zeros(m, dtype=np.int32)
        for i in range(N // m):
            r = (i * m) % m  # this is just i%m... wait, index is r + m*i
        # Actually: for residue r, covered positions are r, r+m, r+2m, ...
        # Count uncovered at those positions
        for r in range(m):
            idx = r + m * np.arange(N // m)
            cnt = (1 - covered[idx]).sum()
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        idx = best_r + m * np.arange(N // m)
        covered[idx] = 1
    return classes

def _local_search(classes, covered, max_time=8.0):
    """Try to improve by changing residues of existing classes or swapping moduli."""
    start = time.time()
    classes = [list(c) for c in classes]
    best_score = _score(classes)
    
    # Precompute for current classes
    for it in range(50):
        if time.time() - start > max_time:
            break
        improved = False
        # Try changing residue for each class
        for i, (m, r) in enumerate(classes):
            # Current contribution
            cur_idx = r + m * np.arange(N // m)
            # Try all residues, find best
            best_r = r
            best_gain = 0
            # Compute coverage count per residue
            # covered array
            for nr in range(m):
                idx = nr + m * np.arange(N // m)
                gain = (1 - covered[idx]).sum() - (covered[cur_idx]).sum() + (covered[idx] & (1-covered[cur_idx])).sum()
                # Simpler: new_uncovered_after = uncovered - covered_by_new + covered_by_old_that_would_lose
                # Let's just compute directly:
                temp_covered = covered.copy()
                # Remove old
                temp_covered[cur_idx] = np.where(temp_covered[cur_idx] == 1, 0, 1)  # this is wrong, need count
            # This is too slow. Use a different approach.
            break
        break
    return classes, best_score

def _try_swap(classes, covered):
    """Try swapping two classes' residues to improve coverage."""
    # Placeholder - too complex for time budget
    return classes, _score(classes)

def construct():
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with greedy
    classes = _greedy()
    score = _score(classes)
    
    # Try to improve via random restarts with local search
    best_classes = [list(c) for c in classes]
    best_score = score
    
    # Try different orderings of moduli
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random ordering of moduli, greedy assignment
        order = DIVISORS[:]
        rng.shuffle(order)
        covered = np.zeros(N, dtype=np.uint8)
        cls = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                idx = r + m * np.arange(N // m)
                cnt = (1 - covered[idx]).sum()
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            cls.append([m, best_r])
            idx = best_r + m * np.arange(N // m)
            covered[idx] = 1
        s = _score(cls)
        if s < best_score:
            best_score = s
            best_classes = [list(c) for c in cls]
            if best_score == 0:
                break
        # Also try local improvement on best
        if time.time() < deadline - 5:
            # Try perturbing best_classes
            perturb = [list(c) for c in best_classes]
            for _ in range(3):
                idx_pert = rng.randrange(len(perturb))
                m_pert = perturb[idx_pert][0]
                perturb[idx_pert][1] = rng.randrange(m_pert)
            s_pert = _score(perturb)
            if s_pert < best_score:
                best_score = s_pert
                best_classes = [list(c) for c in perturb]
                if best_score == 0:
                    break
    
    return best_classes

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Score: {_score(result)}, Classes: {len(result)}")
