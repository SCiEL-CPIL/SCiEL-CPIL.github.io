"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import permutations

# EVOLVE-BLOCK-START
def _build_recursive_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _get_all_permutations(n):
    """Generate all permutations of 1..n as tuples."""
    return set(permutations(range(1, n + 1)))

def _extract_windows(word, n):
    """Extract all length-n windows from word as tuples."""
    windows = set()
    for i in range(len(word) - n + 1):
        windows.add(tuple(word[i:i + n]))
    return windows

def _try_remove_symbol(word, pos, n):
    """
    Try to remove symbol at position pos.
    Returns (new_word, removed_count) if removal is valid, else (None, 0).
    Only valid if removing pos doesn't destroy any required permutation.
    """
    if pos < 0 or pos >= len(word):
        return None, 0
    
    # Check if any required permutation is only present at this position
    # A window starting at i covers positions i..i+n-1
    # Position pos is covered by windows starting at max(0, pos-n+1) to min(pos, len(word)-n)
    
    # First, get all windows that include position pos
    start_min = max(0, pos - n + 1)
    start_max = min(pos, len(word) - n)
    
    # For each such window, check if it appears elsewhere
    # This is expensive, so we do a quick check: is this position critical?
    
    # Quick heuristic: if the symbol at pos equals the symbol at pos+1, 
    # removing pos might be safe (common in superperms)
    # But we need to verify all permutations are still present.
    
    # For efficiency, only attempt removal if it looks promising
    # Check if adjacent symbols are the same
    if pos > 0 and word[pos] == word[pos-1]:
        # Try removing pos-1 instead (dedup consecutive)
        pass
    if pos < len(word) - 1 and word[pos] == word[pos+1]:
        # Could remove either pos or pos+1
        pass
    
    # Actually, let's just do a full check for candidate positions
    # This is O(n * len(word)) per removal attempt
    new_word = word[:pos] + word[pos+1:]
    if len(new_word) < n:
        return None, 0
    
    # Verify all permutations are still present
    # Optimization: only check if the number of unique windows didn't decrease below n!
    new_windows = set()
    for i in range(len(new_word) - n + 1):
        new_windows.add(tuple(new_word[i:i + n]))
    
    # We need all n! permutations
    # Quick count check first
    if len(new_windows) < _get_all_permutations(n).__len__():
        return None, 0
    
    # Full check: all required perms must be in new_windows
    required = _get_all_permutations(n)
    if not required.issubset(new_windows):
        return None, 0
    
    return new_word, 1

def _optimize_word(word, n, time_limit):
    """
    Greedily try to remove symbols to shorten the word while keeping it valid.
    Uses a time limit to avoid running too long.
    """
    start_time = time.time()
    required_perms = _get_all_permutations(n)
    num_perms = len(required_perms)
    
    current_word = word[:]
    removed_count = 0
    
    # Multiple passes: keep trying until no more removals possible or time runs out
    while True:
        if time.time() - start_time > time_limit:
            break
        
        progress = False
        # Try removing each position (in order)
        for pos in range(len(current_word) - 1, -1, -1):  # Backwards so indices stay valid
            if time.time() - start_time > time_limit:
                break
            
            # Quick check: can we remove this position?
            # Only attempt if it might be safe
            # Heuristic: positions where word[pos] == word[pos-1] or word[pos] == word[pos+1]
            # are more likely to be removable
            
            # Actually, let's just try all positions but with a fast pre-check
            # Pre-check: if removing this position would make the window at this position
            # invalid, skip it. But that's complex.
            
            # Try the removal
            new_word, removed = _try_remove_symbol(current_word, pos, n)
            if new_word is not None:
                current_word = new_word
                removed_count += 1
                progress = True
                # Continue trying more removals
                break  # Restart the loop since word changed
        
        if not progress:
            break
    
    return current_word, removed_count

def construct(n):
    # Build the standard recursive superpermutation
    word = _build_recursive_superperm(n)
    
    # Time budget: leave 10 seconds for safety
    start_time = time.time()
    time_limit = 110.0  # seconds
    
    # Attempt to shorten the word
    if n <= 7:  # Only attempt optimization for reasonable n
        optimized_word, removed = _optimize_word(word, n, time_limit)
        if removed > 0:
            word = optimized_word
    
    # Convert to string
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
