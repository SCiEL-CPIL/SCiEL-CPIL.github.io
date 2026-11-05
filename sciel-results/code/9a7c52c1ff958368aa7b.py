# EVOLVE-BLOCK-START
import time
import random

N = 192
MAXC = 48
DEADLINE = 85.0  # wall-clock safety margin under 90s kill

def _is_valid_full(selected):
    """Full validation of a set (independent of construction order)."""
    n = len(selected)
    for idx, a in enumerate(selected):
        others = [selected[t] for t in range(n) if t != idx]
        m = len(others)
        if m < 3:
            continue
        # Check if any 3 distinct others sum to a multiple of a
        # Use residues: for fixed a, we need 3 distinct others with sum % a == 0
        # For each pair (i, j), we need k such that (others[i] + others[j] + others[k]) % a == 0
        # => others[k] % a == (-others[i] - others[j]) % a
        # And k must be distinct from i, j.
        
        # Precompute residues of others
        residues = [x % a for x in others]
        
        # For each residue value r, keep track of indices that have that residue
        # But we need distinct indices, so we need to be careful.
        # Simple approach: for each pair (i, j), compute needed residue, 
        # then check if any k != i, j has that residue.
        
        # To make this faster, group indices by residue
        from collections import defaultdict
        residue_indices = defaultdict(list)
        for idx_o, res in enumerate(residues):
            residue_indices[res].append(idx_o)
        
        for i in range(m):
            for j in range(i + 1, m):
                needed = (-residues[i] - residues[j]) % a
                # Find k != i, j with residues[k] == needed
                if needed in residue_indices:
                    for k in residue_indices[needed]:
                        if k != i and k != j:
                            return False
    return True

def _candidate_ok(x, selected):
    """Check if adding x keeps property: no element a in (selected + [x]) divides sum of 3 distinct others in that set."""
    newset = list(selected) + [x]
    n = len(newset)
    
    # 1) x must not divide any 3-sum of distinct elements from `selected` (others)
    sel = selected
    m = len(sel)
    if m >= 3:
        # Check if x divides any 3-sum of distinct elements from sel
        residues = [s % x for s in sel]
        from collections import defaultdict
        residue_indices = defaultdict(list)
        for idx_s, res in enumerate(residues):
            residue_indices[res].append(idx_s)
        
        for i in range(m):
            for j in range(i + 1, m):
                needed = (-residues[i] - residues[j]) % x
                if needed in residue_indices:
                    for k in residue_indices[needed]:
                        if k != i and k != j:
                            return False
    
    # 2) each a in selected must not divide any 3-sum of distinct elements from newset \ {a}
    for idx_a, a in enumerate(selected):
        others = [newset[t] for t in range(n) if t != idx_a]
        m_o = len(others)
        if m_o < 3:
            continue
        
        residues = [o % a for o in others]
        from collections import defaultdict
        residue_indices = defaultdict(list)
        for idx_o, res in enumerate(residues):
            residue_indices[res].append(idx_o)
        
        for i in range(m_o):
            for j in range(i + 1, m_o):
                needed = (-residues[i] - residues[j]) % a
                if needed in residue_indices:
                    for k in residue_indices[needed]:
                        if k != i and k != j:
                            return False
    return True

def _greedy(order, start_set=None, deadline=None):
    selected = list(start_set) if start_set else []
    present = set(selected)
    for x in order:
        if x in present:
            continue
        if deadline is not None and time.time() > deadline:
            break
        if _candidate_ok(x, selected):
            selected.append(x)
            present.add(x)
    return selected

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Strategy: large numbers first (safe), then small.
    # Try several orderings to maximize cardinality.
    best = []
    
    # Base: all numbers 100..192 are very likely mutually safe. Verify quickly.
    base_candidates = list(range(100, N + 1))
    base = []
    for x in base_candidates:
        if _candidate_ok(x, base):
            base.append(x)
    
    # Now fill with small numbers in various orders.
    small = list(range(1, 100))
    orders = [
        small[::-1],   # descending small
        small,          # ascending small
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        res = _greedy(order, start_set=list(base), deadline=deadline)
        if len(res) > len(best):
            best = res
    
    # Also try starting from scratch with different orders (no base) in case base hurts.
    for order in [list(range(N, 0, -1)), list(range(1, N + 1))]:
        if time.time() > deadline:
            break
        res = _greedy(order, start_set=None, deadline=deadline)
        if len(res) > len(best):
            best = res
    
    # Local improvement: try to add any remaining number not in best.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        present = set(best)
        for x in range(1, N + 1):
            if x in present:
                continue
            if _candidate_ok(x, best):
                best.append(x)
                present.add(x)
                changed = True
    
    # Final safety: verify validity; if invalid, fall back to a known-good small set.
    if not _is_valid_full(best):
        # Fallback: return large numbers only (very safe)
        fb = []
        for x in range(N, 0, -1):
            if _candidate_ok(x, fb):
                fb.append(x)
        if _is_valid_full(fb):
            best = fb
        else:
            # last resort: just large range
            best = [x for x in range(120, N + 1)]
            if not _is_valid_full(best):
                best = list(range(150, N + 1))
    
    return best
# EVOLVE-BLOCK-END
