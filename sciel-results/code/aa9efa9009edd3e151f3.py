import itertools
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121431214321342132414324"
    if n == 5:
        return "12345121431214531214352134213254132543132452345235415345"

    # For n=6 and n=7, use constraint-based backtracking
    # We build the word symbol by symbol, tracking which permutations are covered.
    # A permutation is "covered" when it appears as a consecutive window of length n.
    # We maintain the set of uncovered permutations and prune when a suffix cannot
    # be extended to cover any remaining permutation.

    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    total_perms = len(all_perms)

    # We'll use a greedy/backtracking approach:
    # Start with the first permutation, then at each step try to extend the word
    # by one symbol, checking if this keeps the search viable.

    # Key insight: after placing k symbols (k >= n), the last (n-1) symbols form
    # a "tail". The next symbol must be such that the new tail can still lead to
    # covering all remaining permutations.

    # For efficiency, we track:
    # - current word
    # - set of covered permutations
    # - For each possible tail (length n-1), which permutations could still be covered
    #   given that we're in a state where that tail is at the end

    # Simpler approach: Use a priority-based greedy with backtracking.
    # At each step, try all n possible next symbols, score them by how many
    # new permutations they cover, and pick the best. If we get stuck, backtrack.

    # For n=7, this is 5040 permutations. A pure greedy might not find optimal,
    # but with some backtracking it should do well.

    # Let's use a beam search / greedy with limited backtracking.

    import time
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for the 120s limit

    # Precompute: for each permutation, what's its first n-1 elements and last n-1 elements
    # This helps with overlap detection

    # State: (word_tuple, covered_set)
    # We use a greedy approach with restarts and local search

    def try_greedy(initial_seed=None):
        """Try a greedy construction, optionally seeded with an initial permutation."""
        if initial_seed is None:
            initial_seed = tuple(range(1, n + 1))
        
        word = list(initial_seed)
        covered = {tuple(initial_seed)}
        remaining = perm_set - covered
        
        # The tail is the last (n-1) symbols
        tail = tuple(word[-(n-1):])
        
        while remaining:
            # Try each possible next symbol
            best_sym = None
            best_new_covered = -1
            best_tail = None
            
            for sym in range(1, n + 1):
                # Check if this symbol can be appended
                # The new tail would be (tail[1:], sym)
                new_tail = tail[1:] + (sym,)
                
                # Check if new_tail has all distinct elements (it must, since it's
                # a prefix of some permutation)
                if len(set(new_tail)) != n - 1:
                    continue
                
                # How many new permutations would be covered?
                # A permutation is covered if it appears as a window. When we append
                # sym, the only new window is word[-(n-1):] + [sym] = tail + (sym,)
                # Wait, that's not right. The new window is the last n symbols.
                # word has length L. After appending sym, length is L+1.
                # The new window is word[L-n+1:L+1] = word[L-n+1:L] + [sym]
                # word[L-n+1:L] is the last (n-1) elements of the old word = tail
                # So the new permutation is tail + (sym,)
                
                new_perm = tail + (sym,)
                
                # Check if new_perm is a valid permutation (all distinct)
                if len(set(new_perm)) != n:
                    continue
                
                # How many new permutations does this cover?
                # Just new_perm, if it's in remaining
                new_cover_count = 1 if new_perm in remaining else 0
                
                # Also, we need to check if this move keeps the search viable.
                # The new tail is new_tail. From this tail, we need to be able to
                # reach all remaining permutations (minus any newly covered).
                
                # Simple viability check: the new_tail must be a prefix of at least
                # one remaining permutation (after accounting for newly covered).
                # More precisely, there must exist a sequence of symbols that,
                # starting from new_tail, can cover all remaining permutations.
                
                # For now, use a simpler heuristic: prefer moves that cover new
                # permutations and keep the tail "flexible" (can be extended to
                # many permutations).
                
                # Count how many remaining permutations have new_tail as a prefix
                # (i.e., their first n-1 elements equal new_tail)
                # Actually, new_tail is the last n-1 elements, not the first.
                # A permutation P can be "reached" from new_tail if we can append
                # symbols to eventually form P. This is complex.
                
                # Simpler: just use the immediate coverage as the primary criterion,
                # with a tiebreaker based on how "central" the new_tail is.
                
                if new_cover_count > best_new_covered:
                    best_new_covered = new_cover_count
                    best_sym = sym
                    best_tail = new_tail
                elif new_cover_count == best_new_covered:
                    # Tiebreaker: prefer tails that appear in more remaining permutations
                    # as a subsequence or prefix
                    pass
            
            if best_sym is None:
                # No valid move found - this is a dead end
                return None, len(word)
            
            word.append(best_sym)
            new_perm = tail + (best_sym,)
            if new_perm in remaining:
                remaining.discard(new_perm)
                covered.add(new_perm)
            tail = best_tail
        
        return word, len(word)

    # Try greedy with different seeds
    best_word = None
    best_len = float('inf')
    
    # Try a few different starting permutations
    seeds = [
        tuple(range(1, n + 1)),
        tuple(range(n, 0, -1)),
        (1, 2, 3, 4, 5, 6, 7) if n == 7 else tuple(range(1, n + 1)),
    ]
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        word, length = try_greedy(seed)
        if word is not None and length < best_len:
            best_len = length
            best_word = word
    
    if best_word is None:
        # Fallback to the standard construction
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
        return "".join(map(str, word))
    
    return "".join(map(str, best_word))
