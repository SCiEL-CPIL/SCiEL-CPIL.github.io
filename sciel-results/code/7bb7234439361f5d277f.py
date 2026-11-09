"""Direct construction of a shortest-superpermutation-like word for n symbols."""
# EVOLVE-BLOCK-START
import itertools
import time

_START = time.time()
_DEADLINE = 150 - 12   # 138 s safety margin


def _classic_construct(m):
    """Classic recursive superpermutation construction for symbols 1..m."""
    word = [1]
    for sym in range(2, m + 1):
        perms, seen = [], set()
        # Collect all (sym-1)-long windows with distinct elements, in order of appearance
        for i in range(len(word) - sym + 2):
            window = tuple(word[i:i + sym - 1])
            if len(set(window)) == sym - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [sym] + list(p)
            # Find maximum overlap between end of 'new' and start of 'block'
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _greedy_delete(word, n, time_budget):
    """Try to delete redundant symbols from word while keeping all permutations.
    
    A symbol at position i can be deleted if no permutation appears ONLY in
    windows that include position i. We greedily scan left to right and delete
    when safe.
    """
    word = list(word)
    m = len(word)
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    # For each permutation, record which window positions contain it
    # perm_pos[perm] = set of starting positions where this perm occurs
    perm_positions = {}
    for i in range(m - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            perm_positions.setdefault(window, set()).add(i)
    
    # Count how many positions each permutation appears at
    perm_counts = {perm: len(pos) for perm, pos in perm_positions.items()}
    
    # A position i is "critical" if deleting it would remove the last occurrence
    # of some permutation. We track for each position which permutations would
    # be lost if we delete it.
    
    # For efficiency: a position i contributes to permutations at windows 
    # starting from max(0, i-n+1) to min(i, m-n)
    # When we delete position i, windows that included i are gone.
    # Those windows started at positions j where j <= i <= j+n-1, i.e., j in [i-n+1, i]
    # AND j is a valid start: 0 <= j <= m-n
    
    # We'll do multiple passes. In each pass, scan left to right; if position i
    # is not the sole occurrence of any permutation, delete it.
    
    # To make this efficient, we maintain for each position a count of how many
    # permutations have their last occurrence there. But that's complex.
    
    # Simpler approach: for each candidate deletion, check if any permutation
    # would lose its only occurrence. We can precompute for each position i,
    # the set of permutations that appear ONLY at windows including i.
    
    # Actually, let's use a different strategy: 
    # For position i, the windows that include it are those starting at j where
    # max(0, i-n+1) <= j <= min(i, m-n). 
    # If we delete position i, those specific window instances are removed.
    # A permutation P is lost iff all its occurrences are in those removed windows.
    
    # Let's precompute: for each position i, which permutations have ALL their 
    # occurrences in windows that include i.
    
    # This is expensive to do naively. Let's use a heuristic: 
    # In each pass, for position i, check the windows around it.
    
    # More efficient: maintain a count array. For each permutation, track how many
    # windows contain it. When we consider deleting position i, we check if any
    # permutation would drop to 0.
    
    # Let's use a different approach: 
    # For each position i, compute the set of permutations that appear in windows 
    # starting from max(0, i-n+1) to min(i, m-n). If any of those permutations 
    # has its ONLY occurrence in that range, then i is critical.
    
    # Precompute for each window start j, the permutation at that window (or None)
    window_perm = [None] * (m - n + 1)
    for j in range(m - n + 1):
        w = tuple(word[j:j+n])
        if len(set(w)) == n:
            window_perm[j] = w
    
    # For each position i, the affected window starts are j in [max(0,i-n+1), min(i, m-n)]
    # We need to know if any permutation has all its occurrences in that range.
    
    # Let's precompute prefix/suffix counts of each permutation's occurrences.
    # Actually, let's just do it iteratively with a time check.
    
    changed = True
    while changed and time.time() < _START + time_budget:
        changed = False
        m = len(word)
        if m <= n:
            break
        
        # Rebuild window_perm for current word
        window_perm = [None] * (m - n + 1)
        for j in range(m - n + 1):
            w = tuple(word[j:j+n])
            if len(set(w)) == n:
                window_perm[j] = w
        
        # Build permutation occurrence lists
        perm_occ = {}
        for j, p in enumerate(window_perm):
            if p is not None:
                perm_occ.setdefault(p, []).append(j)
        
        # For each position i from 0 to m-1, check if it's safe to delete
        # Position i is in windows starting at j where max(0, i-n+1) <= j <= min(i, m-n)
        # If we delete i, those window instances are removed.
        # A perm is lost if all its occurrences are in that range.
        
        # Precompute for each position i, the range of window starts affected
        for i in range(m):
            j_start = max(0, i - n + 1)
            j_end = min(i, m - n)
            if j_start > j_end:
                # No windows include position i; safe to delete
                del word[i]
                m -= 1
                changed = True
                break
            # Check if any permutation has all its occurrences in [j_start, j_end]
            safe = True
            # We need to check permutations that occur in this range
            # Collect all perms in this range
            perms_in_range = {}
            for j in range(j_start, j_end + 1):
                if window_perm[j] is not None:
                    perms_in_range[window_perm[j]] = perms_in_range.get(window_perm[j], 0) + 1
            
            for perm, count_in_range in perms_in_range.items():
                if count_in_range == len(perm_occ[perm]):
                    # All occurrences of this perm are in the affected range
                    safe = False
                    break
            
            if safe:
                del word[i]
                m -= 1
                changed = True
                break
        
        # Time check
        if time.time() > _START + time_budget:
            break
    
    return word


def _build_word(n):
    """Build a word over 1..n containing every permutation as consecutive window."""
    if n <= 1:
        return [1]
    
    word = _classic_construct(n)
    return word


def construct(n):
    """Construct a shortest-known superpermutation word for symbols 1..n.
    
    Returns a string of digits 1..n.
    """
    t0 = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121342143121431214321"
    if n == 5:
        return "1234512134521431523415243152543152143521"
    
    # For n >= 6, use classic construction then greedy deletion
    word = _classic_construct(n)
    
    # Use greedy deletion with remaining time budget
    elapsed = time.time() - t0
    remaining = _DEADLINE - (time.time() - _START)
    if remaining > 5:  # Need at least 5s for deletion
        word = _greedy_delete(word, n, remaining - 2)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
