"""Hierarchical cell-decomposition approach to odd covering systems."""
import time
import random
import math
from itertools import product

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Prime powers: 27, 25, 49, 121
PP = [27, 25, 49, 121]
PP_NAMES = ['p27', 'p25', 'p49', 'p121']

# All odd divisors > 1
_DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
_DIVISORS_SET = set(_DIVISORS)
_PP_SET = set(PP)

# Classify divisors
_PRIME_POWERS = [m for m in _DIVISORS if m in _PP_SET]  # 27, 25, 49, 121
_COMPOSITES = [m for m in _DIVISORS if m not in _PP_SET]
_COMPOSITES_SORTED = sorted(_COMPOSITES, key=lambda m: BASE // m, reverse=True)


def _fast_uncovered(classes):
    """Count residues mod BASE covered by no class."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _build_covered(classes):
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return covered


def _cell_index(x):
    """Map residue x in [0, BASE) to cell index (i27, i25, i49, i121)."""
    # x mod 27, x mod 25, x mod 49, x mod 121
    # But these are not independent in a simple way for arbitrary x.
    # However, for a FIXED assignment of residues (r27, r25, r49, r121),
    # a residue x is covered by the anchor iff x % 27 == r27 AND x % 25 == r25 AND ...
    # Wait, that's not right. The anchors are SEPARATE classes.
    # Class (27, r27) covers {x : x % 27 == r27}
    # Class (25, r25) covers {x : x % 25 == r25}
    # A residue x is covered by at least one anchor if (x%27==r27) or (x%25==r25) or (x%49==r49) or (x%121==r121)
    # The "uncovered by anchors" set is {x : x%27!=r27 and x%25!=r25 and x%49!=r49 and x%121!=r121}
    # This is the product of the complements. By CRT, for each combination of residues
    # (a27 in [0,26]\{r27}, a25 in [0,24]\{r25}, a49 in [0,48]\{r49}, a121 in [0,120]\{r121}),
    # there is exactly one residue x mod 51975 satisfying x%27=a27, x%25=a25, x%49=a49, x%121=a121.
    # So the number of residues uncovered by anchors is 26*24*48*120.
    # And each such "cell" (a27, a25, a49, a121) corresponds to exactly one residue mod BASE.
    pass


def _anchor_uncovered_mask(r27, r25, r49, r121):
    """Return a bytearray of length BASE, 1 where NOT covered by any anchor."""
    # A residue x is uncovered by anchors iff:
    #   x % 27 != r27 AND x % 25 != r25 AND x % 49 != r49 AND x % 121 != r121
    # We can compute this directly.
    uncovered = bytearray(BASE)
    for x in range(BASE):
        if x % 27 != r27 and x % 25 != r25 and x % 49 != r49 and x % 121 != r121:
            uncovered[x] = 1
    return uncovered


def _composite_covers_uncovered(m, r, uncovered_mask):
    """Count how many residues in class (m, r) are uncovered by anchors."""
    count = 0
    for x in range(r, BASE, m):
        if uncovered_mask[x]:
            count += 1
    return count


def _build_solution_from_anchors(r27, r25, r49, r121, time_limit=5.0):
    """Given anchor residues, greedily assign composites to maximize uncovered coverage."""
    start_time = time.time()
    
    # Compute which residues are uncovered by anchors
    uncovered = bytearray(BASE)
    for x in range(BASE):
        if x % 27 != r27 and x % 25 != r25 and x % 49 != r49 and x % 121 != r121:
            uncovered[x] = 1
    
    initial_uncovered = sum(uncovered)
    
    # Greedily add composites
    covered = bytearray(BASE)
    # Mark anchor-covered residues
    for x in range(r27, BASE, 27):
        covered[x] = 1
    for x in range(r25, BASE, 25):
        covered[x] = 1
    for x in range(r49, BASE, 49):
        covered[x] = 1
    for x in range(r121, BASE, 121):
        covered[x] = 1
    
    classes = [(27, r27), (25, r25), (49, r49), (121, r121)]
    
    # For each composite, find best residue
    for m in _COMPOSITES_SORTED:
        if time.time() - start_time > time_limit:
            break
        best_r = 0
        best_gain = -1
        # For small moduli, try all residues; for large, sample
        if m <= 100:
            candidates = range(m)
        else:
            candidates = list(range(m))  # still try all for now
            if m > 500:
                candidates = [random.randint(0, m - 1) for _ in range(100)] + list(range(m))[:50]
        
        for r in candidates:
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    
    score = BASE - sum(covered)
    return classes, score


def _local_search_cell(classes, time_limit=5.0):
    """Local search that works on the cell structure."""
    start_time = time.time()
    best_classes = list(classes)
    best_score = _fast_uncovered(best_classes)
    
    if best_score == 0:
        return best_classes, 0
    
    covered = _build_covered(best_classes)
    mod_to_res = {m: r for m, r in best_classes}
    used_mods = set(mod_to_res.keys())
    available_mods = [m for m in _DIVISORS if m not in used_mods]
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try changing residues
        for m in list(used_mods):
            old_r = mod_to_res[m]
            for x in range(old_r, BASE, m):
                covered[x] = 0
            
            best_new_r = old_r
            best_new_score = best_score
            
            if m <= 80:
                candidates = range(m)
            else:
                candidates = [random.randint(0, m - 1) for _ in range(80)]
            
            for r in candidates:
                if r == old_r:
                    continue
                for x in range(r, BASE, m):
                    covered[x] = 1
                score = BASE - sum(covered)
                if score < best_new_score:
                    best_new_score = score
                    best_new_r = r
                for x in range(r, BASE, m):
                    covered[x] = 0
            
            if best_new_r != old_r and best_new_score < best_score:
                for x in range(best_new_r, BASE, m):
                    covered[x] = 1
                mod_to_res[m] = best_new_r
                best_score = best_new_score
                improved = True
            else:
                for x in range(old_r, BASE, m):
                    covered[x] = 1
        
        # Try adding a modulus
        if available_mods:
            sample_size = min(8, len(available_mods))
            for m in random.sample(available_mods, sample_size):
                best_r = 0
                best_gain = 0
                if m <= 80:
                    r_range = range(m)
                else:
                    r_range = [random.randint(0, m - 1) for _ in range(50)]
                for r in r_range:
                    gain = 0
                    for x in range(r, BASE, m):
                        if not covered[x]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                if best_gain > 0:
                    for x in range(best_r, BASE, m):
                        covered[x] = 1
                    new_score = BASE - sum(covered)
                    if new_score < best_score:
                        used_mods.add(m)
                        mod_to_res[m] = best_r
                        available_mods.remove(m)
                        best_score = new_score
                        improved = True
                    else:
                        for x in range(best_r, BASE, m):
                            covered[x] = 0
        
        # Try removing
        for m in list(used_mods):
            r = mod_to_res[m]
            for x in range(r, BASE, m):
                covered[x] = 0
            new_score = BASE - sum(covered)
            if new_score < best_score:
                used_mods.discard(m)
                del mod_to_res[m]
                available_mods.append(m)
                best_score = new_score
                improved = True
            else:
                for x in range(r, BASE, m):
                    covered[x] = 1
        
        if not improved:
            no_improve += 1
            if no_improve >= 10:
                break
        else:
            no_improve = 0
        
        if best_score == 0:
            break
    
    best_classes = [[m, mod_to_res[m]] for m in sorted(used_mods)]
    return best_classes, best_score


def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    best_classes = None
    best_score = BASE  # worst case
    
    # Phase 1: Try several anchor configurations
    # There are 27*25*49*121 = 4,041,825 possible anchor configs.
    # We can't try them all, so we sample strategically.
    
    # First, try a few random anchor configs
    n_anchor_trials = 30
    for trial in range(n_anchor_trials):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random anchor residues
        r27 = random.randint(0, 26)
        r25 = random.randint(0, 24)
        r49 = random.randint(0, 48)
        r121 = random.randint(0, 120)
        
        classes, score = _build_solution_from_anchors(r27, r25, r49, r121, time_limit=3.0)
        
        if score < best_score:
            best_score = score
            best_classes = classes
        
        if time.time() - start_time > DEADLINE:
            break
    
    if best_classes is None:
        # Fallback: greedy
        covered = bytearray(BASE)
        classes = []
        for m in sorted(_DIVISORS, reverse=True):
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = 0
                for x in range(r, BASE, m):
                    if not covered[x]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
        best_classes = classes
        best_score = _fast_uncovered(best_classes)
    
    # Phase 2: Local search on the best solution
    if time.time() - start_time < DEADLINE:
        remaining = DEADLINE - (time.time() - start_time)
        classes, score = _local_search_cell(best_classes, time_limit=min(10.0, remaining))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Phase 3: More anchor trials with local search
    while time.time() - start_time < DEADLINE:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        r27 = random.randint(0, 26)
        r25 = random.randint(0, 24)
        r49 = random.randint(0, 48)
        r121 = random.randint(0, 120)
        
        classes, score = _build_solution_from_anchors(r27, r25, r49, r121, time_limit=2.0)
        
        if score < best_score:
            # Local search to refine
            if time.time() - start_time < DEADLINE:
                rem2 = DEADLINE - (time.time() - start_time)
                classes, score = _local_search_cell(classes, time_limit=min(5.0, rem2))
            
            if score < best_score:
                best_score = score
                best_classes = classes
        
        if best_score == 0:
            break
    
    return [[m, r] for m, r in best_classes]
