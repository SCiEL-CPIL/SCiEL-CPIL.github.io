"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import factorial

def construct(n):
    """Construct a superpermutation word over symbols 1..n.
    
    For n <= 4, use the optimal recursive construction.
    For n >= 5, use simulated annealing on the ordering of permutation blocks.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # For small n, the classic recursive construction is optimal
    if n <= 4:
        return _recursive_superperm(n)
    
    # For n >= 5, use simulated annealing
    # The idea: we have n! permutations. Each permutation is a block of length n.
    # We need to order them so that the total concatenated length is minimized.
    # The overlap between consecutive permutations determines the length reduction.
    
    # Precompute all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # For n=7, num_perms = 5040, which is manageable
    
    # Compute overlap matrix: overlap[i][j] = maximum k such that suffix of perms[i] of length k
    # equals prefix of perms[j] of length k. This determines how much we can save by placing j after i.
    # overlap[i][j] = max k in [0, n-1] such that perms[i][n-k:] == perms[j][:k]
    
    # We want to find a Hamiltonian path that maximizes total overlap (minimizes total length).
    # Total length = n * num_perms - sum of overlaps along the path
    
    # For n=7, 5040 permutations, computing the full 5040x5040 overlap matrix is feasible.
    # Each overlap computation is O(n), so total is O(n * (n!)^2) = O(7 * 5040^2) ≈ 178M ops.
    # This might be too slow in Python. Let's think of a smarter approach.
    
    # Actually, for n=7, 5040^2 = ~25.4M pairs, each with up to 7 comparisons.
    # That's ~178M character comparisons. In Python this could take a few seconds.
    # Let's try it.
    
    # Compute overlap matrix
    # overlap[i][j] = max k in 0..n-1 such that perms[i][n-k:] == perms[j][:k]
    
    # To speed up, we can use the fact that if perms[i][n-k:] == perms[j][:k],
    # then the last n-k elements of perms[i] must equal the first k elements of perms[j].
    
    # Let's compute this efficiently.
    
    # Precompute for each permutation, its suffixes and prefixes as tuples
    # suffixes[i][k] = perms[i][n-k:] for k in 0..n-1
    # prefixes[j][k] = perms[j][:k] for k in 0..n-1
    
    # Then overlap[i][j] = max k such that suffixes[i][k] == prefixes[j][k]
    
    # But storing all these is memory intensive. Let's just compute on the fly but cache.
    
    # Actually, let's just compute the overlap matrix directly.
    
    # For n=7, this should be feasible.
    
    # Build a mapping from tuple to index for quick lookup
    # We want to find, for each pair (i, j), the maximum overlap.
    
    # Alternative approach: For each permutation i, and for each possible overlap length k (from n-1 down to 0),
    # check if the suffix of perms[i] of length k matches the prefix of perms[j] of length k.
    # We can precompute a dictionary: for each k, map prefix of length k to list of permutations that have that prefix.
    
    # Let's use a different strategy: 
    # For each permutation j, and for each k from n-1 down to 0,
    # the prefix of length k is perms[j][:k]. We need to find permutations i such that perms[i][n-k:] == perms[j][:k].
    # This means the suffix of i of length k equals the prefix of j of length k.
    
    # Let's precompute: for each k, a dictionary mapping suffix of length k to list of permutation indices.
    
    # Actually, let's just compute the overlap matrix in a reasonable time.
    
    # For n=7, let's estimate: 5040*5040 = 25,401,600 pairs. For each pair, we check up to 6 overlaps (k=6 down to 1, plus k=0 always works).
    # Average maybe 3-4 comparisons per pair. So ~100M comparisons. In Python, this might take 30-60 seconds.
    # That's too slow. Let's optimize.
    
    # Better approach: For each permutation i, compute its suffixes of all lengths.
    # For each permutation j, compute its prefixes of all lengths.
    # Then for each (i, j), find the max k where suffix_i[k] == prefix_j[k].
    
    # We can precompute:
    # suffix_map[k][tuple] = list of i such that perms[i][n-k:] == tuple
    # prefix_map[k][tuple] = list of j such that perms[j][:k] == tuple
    
    # Then overlap[i][j] >= k iff perms[i][n-k:] == perms[j][:k]
    # So overlap[i][j] = max k such that (perms[i][n-k:], perms[j][:k]) match.
    
    # Let's build the overlap matrix using this approach.
    
    # For k from 1 to n-1:
    #   For each permutation i, compute suffix = perms[i][n-k:]
    #   For each permutation j, compute prefix = perms[j][:k]
    #   If suffix == prefix, then overlap[i][j] = max(overlap[i][j], k)
    
    # This is still O(n * (n!)^2) but with smaller constant.
    
    # Actually, let's just do it directly but with early termination.
    
    # Let's try a different approach: use the structure of permutations.
    # Two permutations i and j have overlap k if the last k elements of i are the first k elements of j.
    # This means i[n-k:] = j[:k].
    
    # For n=7, let's just compute it and hope it's fast enough.
    
    # Initialize overlap matrix with 0
    # overlap[i][j] will be the maximum overlap
    
    # Let's use a list of lists for the overlap matrix.
    # For n=7, 5040x5040 = 25.4M entries. Each is a small int. This is ~200MB in Python, which is too much.
    
    # We need a more memory-efficient approach.
    
    # Alternative: don't store the full matrix. Instead, during the annealing, compute overlaps on the fly.
    # But that would be slow.
    
    # Let's think differently. The key insight is that for a random permutation of permutations,
    # the expected overlap is small. Most adjacent pairs will have overlap 0 or 1.
    
    # For the annealing, we only need to evaluate the energy of the current path and proposed paths.
    # The energy is the total length = n * num_perms - sum of overlaps.
    # So we need to compute the sum of overlaps along the path.
    
    # When we swap two adjacent elements in the path, only the overlaps involving those elements change.
    # Specifically, if we swap positions p and p+1, the affected overlaps are:
    # - overlap[path[p-1]][path[p]] (if p > 0)
    # - overlap[path[p]][path[p+1]]
    # - overlap[path[p+1]][path[p+2]] (if p+1 < num_perms-1)
    
    # So we only need to compute a few overlaps per move.
    
    # Let's precompute the overlap function as a callable that takes two permutation indices and returns the overlap.
    
    def compute_overlap(i, j):
        """Compute the maximum overlap between perms[i] and perms[j]."""
        # Max possible overlap is n-1 (can't be n because that would mean they're the same permutation)
        for k in range(n - 1, 0, -1):
            if perms[i][n - k:] == perms[j][:k]:
                return k
        return 0
    
    # For the annealing, we'll maintain the current path and the total overlap.
    # When we propose a swap, we compute the change in total overlap.
    
    # Initial path: random permutation of all permutations
    path = list(range(num_perms))
    random.shuffle(path)
    
    # Compute initial total overlap
    def total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += compute_overlap(path[i], path[i + 1])
        return total
    
    # This is O(num_perms * n) per evaluation, which is 5040 * 7 = 35280 ops. Fine.
    
    current_overlap = total_overlap(path)
    current_length = n * num_perms - current_overlap
    
    best_overlap = current_overlap
    best_path = path[:]
    best_length = current_length
    
    # Simulated annealing
    # Temperature schedule: start high, cool down
    # We'll use a geometric cooling schedule.
    
    t_start = time.time()
    
    # Parameters
    T0 = 10.0  # Initial temperature
    alpha = 0.999  # Cooling rate
    T_min = 0.01
    
    # Number of iterations: we'll run until deadline
    iteration = 0
    
    while time.time() < deadline:
        T = T0 * (alpha ** iteration)
        if T < T_min:
            T = T_min
        
        # Propose a move: swap two random positions
        p = random.randint(0, num_perms - 2)
        q = random.randint(p + 1, num_perms - 1)
        
        # Compute the change in overlap if we swap path[p] and path[q]
        # This is complex because it affects multiple overlaps.
        # For simplicity, let's just swap adjacent elements for now.
        
        # Swap adjacent elements at position p
        a = path[p]
        b = path[p + 1]
        
        # Compute the overlap change
        # Before swap: overlaps involving positions p-1, p, p+1, p+2
        # After swap: overlaps involving the same positions but with a and b swapped
        
        old_overlap = 0
        new_overlap = 0
        
        # Overlap at position p-1 to p
        if p > 0:
            old_overlap += compute_overlap(path[p - 1], a)
            new_overlap += compute_overlap(path[p - 1], b)
        
        # Overlap at position p to p+1
        old_overlap += compute_overlap(a, b)
        new_overlap += compute_overlap(b, a)
        
        # Overlap at position p+1 to p+2
        if p + 1 < num_perms - 1:
            old_overlap += compute_overlap(b, path[p + 2])
            new_overlap += compute_overlap(a, path[p + 2])
        
        delta_overlap = new_overlap - old_overlap
        delta_length = -delta_overlap  # Length decreases when overlap increases
        
        # Accept if delta_length < 0 (improvement) or with probability exp(-delta_length / T)
        if delta_length < 0:
            path[p], path[p + 1] = b, a
            current_overlap += delta_overlap
            current_length += delta_length
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_path = path[:]
                best_length = current_length
        else:
            import math
            prob = math.exp(-delta_length / T)
            if random.random() < prob:
                path[p], path[p + 1] = b, a
                current_overlap += delta_overlap
                current_length += delta_length
                
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = path[:]
                    best_length = current_length
        
        iteration += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Reconstruct the word from the best path
    word = list(perms[best_path[0]])
    for i in range(1, len(best_path)):
        prev = perms[best_path[i - 1]]
        curr = perms[best_path[i]]
        # Find the overlap
        overlap = 0
        for k in range(n - 1, 0, -1):
            if prev[n - k:] == curr[:k]:
                overlap = k
                break
        word.extend(curr[overlap:])
    
    return "".join(map(str, word))


def _recursive_superperm(n):
    """Classic recursive construction of a superpermutation."""
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
# EVOLVE-BLOCK-END
