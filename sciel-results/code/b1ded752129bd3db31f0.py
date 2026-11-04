"""Improved program: greedy set-cover + local search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

# Precompute all words as integers 0..3^7-1
_Q = 3
_N = 7
_RADIUS = 1
_TOTAL = _Q ** _N  # 2187

# Precompute neighbors (distance <= 1) for each word
def _build_neighbors(q, n, radius):
    """Build list of neighbors (distance <= radius) for each word in [0, q^n)."""
    neighbors = []
    for x in range(q ** n):
        # Convert to digits
        digits = []
        tmp = x
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1]  # most significant first
        
        covered = set()
        covered.add(x)
        # All words at distance exactly 1
        for pos in range(n):
            for val in range(q):
                if val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                new_x = 0
                for d in new_digits:
                    new_x = new_x * q + d
                covered.add(new_x)
        neighbors.append(covered)
    return neighbors

def _digits(x, n, q):
    """Convert integer to list of n digits in base q."""
    digits = []
    tmp = x
    for _ in range(n):
        digits.append(tmp % q)
        tmp //= q
    return digits[::-1]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    total = q ** n  # 2187
    
    # Build neighbor sets
    neighbors = _build_neighbors(q, n, radius)
    
    # Greedy set cover
    uncovered = set(range(total))
    selected = []
    
    # For each candidate codeword c, its coverage is neighbors[c]
    # Greedy: pick c that covers most uncovered points
    
    while uncovered:
        if time.time() > deadline - 20.0:
            break
        
        best_c = -1
        best_count = 0
        for c in range(total):
            if not uncovered:
                break
            # Count how many uncovered points c covers
            cnt = 0
            for p in neighbors[c]:
                if p in uncovered:
                    cnt += 1
                    if cnt > best_count:
                        pass
            if cnt > best_count:
                best_count = cnt
                best_c = c
        
        if best_c == -1 or best_count == 0:
            # Shouldn't happen, but break to avoid infinite loop
            break
        
        selected.append(best_c)
        # Remove covered points
        for p in neighbors[best_c]:
            uncovered.discard(p)
    
    # If greedy didn't finish, add more greedily
    while uncovered and time.time() < deadline - 15.0:
        best_c = -1
        best_count = 0
        for c in range(total):
            if c in selected:
                continue
            cnt = 0
            for p in neighbors[c]:
                if p in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        selected.append(best_c)
        for p in neighbors[best_c]:
            uncovered.discard(p)
    
    # Verify coverage and do local search refinement
    def _is_covered(sel):
        """Check if sel covers all points."""
        covered = set()
        for c in sel:
            covered.update(neighbors[c])
        return len(covered) == total
    
    def _coverage_count(sel):
        """Count how many points are covered by sel."""
        covered = set()
        for c in sel:
            covered.update(neighbors[c])
        return len(covered)
    
    # Ensure full coverage
    if not _is_covered(selected):
        # Add more codewords until covered
        while not _is_covered(selected) and time.time() < deadline - 10.0:
            covered = set()
            for c in selected:
                covered.update(neighbors[c])
            uncovered_pts = [p for p in range(total) if p not in covered]
            if not uncovered_pts:
                break
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in selected:
                    continue
                gain = 0
                for p in neighbors[c]:
                    if p not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            selected.append(best_c)
    
    # Local search: try to remove codewords, then try 1-exchanges
    if time.time() < deadline - 5.0:
        # Phase 1: Try removing each codeword
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for i in range(len(selected)):
                candidate = selected[:i] + selected[i+1:]
                if _is_covered(candidate):
                    selected = candidate
                    changed = True
                    break  # Restart removal loop
    
    # Phase 2: 1-exchange moves (remove one, add another)
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break
                # Try removing selected[i] and adding each other c
                sel_without = selected[:i] + selected[i+1:]
                covered_without = set()
                for c in sel_without:
                    covered_without.update(neighbors[c])
                
                # Find best replacement
                best_c = -1
                best_gain = 0
                for c in range(total):
                    if c in selected:
                        continue
                    gain = 0
                    for p in neighbors[c]:
                        if p not in covered_without:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                
                if best_c != -1 and best_gain > 0:
                    # Check if this actually maintains coverage
                    new_sel = sel_without + [best_c]
                    if _is_covered(new_sel):
                        selected = new_sel
                        improved = True
                        break  # Restart exchange loop
    
    # Convert selected integers to list of digit lists
    result = [_digits(c, n, q) for c in selected]
    
    # Deduplicate
    seen = set()
    unique_result = []
    for word in result:
        t = tuple(word)
        if t not in seen:
            seen.add(t)
            unique_result.append(word)
    
    return unique_result
# EVOLVE-BLOCK-END
