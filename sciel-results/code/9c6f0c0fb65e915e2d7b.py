"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _tetracode_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    if q != 3 or n < 4:
        # Fallback: all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _build_distance_graph(q, n, radius):
    """
    Build the distance-radius graph: vertices are all q^n words,
    edges connect words within Hamming distance <= radius.
    Returns (num_vertices, adjacency_list).
    """
    total = q ** n
    # Generate all words as indices 0..total-1
    # Vertex i corresponds to the word represented by base-q digits of i
    
    adj = [set() for _ in range(total)]
    
    # For each vertex, find all vertices within Hamming distance <= radius
    # Using base-q representation
    words = []
    for i in range(total):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(w)
    
    # Build adjacency by checking Hamming distance
    # For efficiency, precompute digit arrays
    digits = np.array(words)  # shape (total, n)
    
    for i in range(total):
        # Find all j with Hamming distance <= radius from i
        # Compare digit by digit
        diffs = np.sum(digits != digits[i], axis=1)
        neighbors = np.where(diffs <= radius)[0]
        adj[i] = set(neighbors.tolist())
    
    return total, adj


def _mis_to_dominating_set(q, n, radius, deadline):
    """
    Find a minimum dominating set using MIS on the complement graph.
    In the distance-radius graph G (edges = within distance <= radius),
    a dominating set D is a set such that every vertex is in D or adjacent to D.
    This is equivalent to: D is a vertex cover of the complement graph... no.
    
    Actually: D is a dominating set in G iff V \\ D is an independent set in G
    (no two vertices in V\\D are adjacent in G, i.e., no two uncovered vertices
    are within distance radius of each other... wait that's not right either).
    
    Let me reconsider: A dominating set D means for every v in V, v in D or
    exists d in D with v adjacent to d (or v=d).
    
    Equivalently: V \\ D is a set where no vertex has a neighbor in D... no.
    V \\ D is a set S such that for every s in S, s has no neighbor in D,
    meaning all neighbors of s are in S. So S is a "closed neighborhood complement".
    
    Actually the standard reduction: D is a dominating set iff V\D is an independent
    set in the graph where we remove self-loops... Let me think more carefully.
    
    D dominates V iff for every v not in D, v has at least one neighbor in D.
    This means: for every v in V\D, N(v) ∩ D ≠ ∅, i.e., N(v) ⊄ V\D.
    Equivalently: V\D contains no vertex v such that N(v) ⊆ V\D.
    That's not simply "independent set".
    
    The correct reduction: D is a dominating set iff V\D is a set with no
    "isolated" vertices in the induced subgraph... no.
    
    Let me use a different approach: greedy local search.
    """
    total = q ** n
    start_time = time.time()
    
    # Generate all words
    words = []
    for i in range(total):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
    
    # Build coverage sets: for each word index, which words does it cover (within radius)?
    # Use numpy for speed
    digits = np.array(words)  # (total, n)
    
    # Precompute coverage: cover[i] = set of indices covered by word i
    cover = []
    for i in range(total):
        diffs = np.sum(digits != digits[i], axis=1)
        covered = set(np.where(diffs <= radius)[0].tolist())
        cover.append(covered)
    
    # Start with tetracode-based construction
    initial = _tetracode_construction(q, n, radius)
    initial_indices = []
    for w in initial:
        # Find index of this word
        idx = 0
        for j, d in enumerate(reversed(w)):
            idx += d * (q ** j)
        initial_indices.append(idx)
    
    # Convert to set
    current = set(initial_indices)
    
    # Compute uncovered
    def get_uncovered(code_set):
        covered = set()
        for idx in code_set:
            covered.update(cover[idx])
        return set(range(total)) - covered
    
    uncovered = get_uncovered(current)
    
    # Local search: try removing vertices, try swapping
    best_code = set(current)
    best_size = len(current)
    
    # Iterative improvement
    while time.time() < deadline:
        improved = False
        
        # Try removing each vertex
        for idx in list(current):
            if time.time() >= deadline:
                break
            if idx in best_code and idx not in current:
                continue
            new_set = current - {idx}
            new_uncovered = get_uncovered(new_set)
            if not new_uncovered:
                # Valid removal
                current = new_set
                uncovered = new_uncovered
                improved = True
                if len(current) < best_size:
                    best_code = set(current)
                    best_size = len(current)
                break  # Restart removal loop
        
        if time.time() >= deadline:
            break
        
        if not improved:
            # Try swaps: remove one, add one that covers more
            best_swap = None
            best_gain = 0
            for idx in list(current):
                if time.time() >= deadline:
                    break
                if idx not in current:
                    continue
                # What does removing idx uncover?
                # Vertices only covered by idx
                only_covered_by_idx = set()
                for v in cover[idx]:
                    if v not in cover[idx]:  # placeholder
                        pass
                # Simpler: compute what's lost
                temp_set = current - {idx}
                temp_uncovered = get_uncovered(temp_set)
                if not temp_uncovered:
                    # Already handled by removal
                    continue
                # Try adding each uncovered vertex
                for add_idx in list(temp_uncovered):
                    if time.time() >= deadline:
                        break
                    if add_idx in current:
                        continue
                    new_set = temp_set | {add_idx}
                    new_uncovered = get_uncovered(new_set)
                    gain = len(temp_uncovered) - len(new_uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (idx, add_idx)
            
            if best_swap and best_gain > 0:
                remove_idx, add_idx = best_swap
                current = (current - {remove_idx}) | {add_idx}
                uncovered = get_uncovered(current)
                improved = True
                if len(current) < best_size:
                    best_code = set(current)
                    best_size = len(current)
        
        if not improved:
            # Try adding a vertex that covers the most uncovered
            if uncovered:
                best_add = None
                best_cover_count = 0
                for v in list(uncovered):
                    if time.time() >= deadline:
                        break
                    new_covered = len(cover[v] & uncovered)
                    if new_covered > best_cover_count:
                        best_cover_count = new_covered
                        best_add = v
                if best_add is not None:
                    current = current | {best_add}
                    uncovered = get_uncovered(current)
                    if len(current) < best_size:
                        best_code = set(current)
                        best_size = len(current)
            else:
                # No uncovered and no improvement found; try random restarts or break
                break
    
    # Convert back to list of words
    result = [list(words[idx]) for idx in best_code]
    return result


def construct(q, n, radius):
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Try MIS/local search approach
    try:
        result = _mis_to_dominating_set(q, n, radius, deadline)
        # Verify validity
        if result:
            # Quick check: ensure all words are covered
            words_arr = np.array(result)
            all_words = np.array(list(itertools.product(range(q), repeat=n)))
            # Check coverage
            covered = np.zeros(q ** n, dtype=bool)
            for w in words_arr:
                diffs = np.sum(all_words != w, axis=1)
                covered |= (diffs <= radius)
            if covered.all():
                return [list(w) for w in result]
    except Exception:
        pass
    
    # Fallback
    return _tetracode_construction(q, n, radius)
# EVOLVE-BLOCK-END
