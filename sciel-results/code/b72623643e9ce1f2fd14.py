# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius r.
    Uses a randomized greedy set cover followed by aggressive local search (redundancy removal).
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Special case handling for small n
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    
    # For the specific task parameters (q=3, n=6, radius=1), we know the optimal is around 73.
    # The direct sum construction gives 9 * 3**(6-4) = 81 words.
    # We aim to beat 81.
    
    # Precompute all words and their neighbors
    # Space size is q**n. For q=3, n=6, size=729.
    # Each word has 1 + n*(q-1) neighbors within distance <= radius (if radius=1).
    # For radius=1, neighbors = 1 + 6*2 = 13.
    
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (indices of words within distance <= radius)
    # neighbors[i] is a set of indices j such that dist(word_i, word_j) <= radius
    # We need this for the greedy set cover and local search.
    
    # Optimization: Only compute neighbors on the fly or precompute if size is small enough.
    # 729 * 13 is small. Let's precompute the "coverage sets" for each potential codeword.
    # coverage_set[i] = set of indices in the space covered by codeword i.
    
    coverage_sets = []
    for w in all_words:
        covered = set()
        # Generate all words within distance <= radius
        # For radius=1, this is efficient.
        # General case: generate all permutations of changes.
        # Since n and q are small, we can iterate through all space points and check distance,
        # but that's O(N^2). Better to generate neighbors directly.
        
        # Efficient neighbor generation for small radius
        if radius == 1:
            covered.add(word_to_idx[w])
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        # Create new word
                        # Since w is a tuple, we need to convert to list, modify, convert back
                        # To speed up, we can use a precomputed mapping or just construct the key
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_w = tuple(new_w)
                        covered.add(word_to_idx[new_w])
        elif radius == 2:
            # More complex, but for this task radius is likely 1.
            # Fallback to distance check if radius > 1 and n*q is small
            for v in all_words:
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    covered.add(word_to_idx[v])
        else:
            # General fallback
            for v in all_words:
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    covered.add(word_to_idx[v])
        
        coverage_sets.append(covered)
    
    # Randomized Greedy Set Cover
    # We want to select a subset S of codewords such that union of coverage_sets[S] == all_indices
    # Greedy: pick codeword that covers the most uncovered points.
    
    def greedy_set_cover():
        uncovered = set(range(num_words))
        selected = []
        
        # To make it randomized, we can shuffle the order of consideration or add noise.
        # Standard greedy is deterministic. Let's try a slightly randomized greedy.
        
        while uncovered:
            best_idx = -1
            best_gain = 0
            # Iterate through all candidates
            # Optimization: Only consider candidates that intersect uncovered
            for i in range(num_words):
                gain = len(coverage_sets[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1:
                # Should not happen if coverage is possible
                break
            
            selected.append(best_idx)
            uncovered -= coverage_sets[best_idx]
        
        return selected
    
    # Local Search: Remove redundant codewords
    # A codeword is redundant if its coverage is fully contained in the union of other selected codewords.
    # i.e., coverage_sets[i] is subset of union(coverage_sets[j] for j in selected, j!=i)
    
    def remove_redundant(selected):
        selected_set = set(selected)
        changed = True
        while changed:
            changed = False
            # Check each selected codeword
            # To speed up, we can check in random order
            for i in list(selected_set):
                # Check if i is redundant
                # Union of others
                # Efficient check: 
                # If there exists a point p in coverage_sets[i] such that p is NOT covered by any other selected codeword, then i is not redundant.
                # Conversely, if for all p in coverage_sets[i], p is covered by some other j in selected_set \ {i}, then i is redundant.
                
                is_redundant = True
                for p in coverage_sets[i]:
                    # Is p covered by any other selected codeword?
                    covered_by_other = False
                    for j in selected_set:
                        if j == i:
                            continue
                        if p in coverage_sets[j]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    selected_set.remove(i)
                    changed = True
        
        return list(selected_set)
    
    # Multi-start ILS
    best_solution = None
    best_size = float('inf')
    
    # Initial solution via greedy
    init_sol = greedy_set_cover()
    init_sol = remove_redundant(init_sol)
    best_solution = init_sol
    best_size = len(init_sol)
    
    current_time = time.time()
    iterations = 0
    
    while current_time < deadline:
        iterations += 1
        # Perturbation: Random k-swap
        # Remove k random codewords, add k random new codewords
        k = random.choice([1, 2, 3])
        if len(best_solution) < k:
            k = len(best_solution)
        
        # Create a copy
        current_sol = list(best_solution)
        
        # Remove k random ones
        to_remove = random.sample(current_sol, k)
        for idx in to_remove:
            current_sol.remove(idx)
        
        # Add k random new ones (from the full space, ensuring they are not already in current_sol)
        # We can pick from all_words indices
        candidates = list(set(range(num_words)) - set(current_sol))
        if not candidates:
            continue
        
        to_add = random.sample(candidates, min(k, len(candidates)))
        current_sol.extend(to_add)
        
        # Local Search: Remove redundant
        current_sol = remove_redundant(current_sol)
        
        # Check validity (should be valid if we started valid and only removed redundant ones? 
        # Wait, adding random codewords might not cover everything if we removed important ones.
        # We must ensure the solution is still a valid cover.
        
        # Check if current_sol covers everything
        covered = set()
        for idx in current_sol:
            covered |= coverage_sets[idx]
        
        if len(covered) < num_words:
            # Invalid cover. We can try to repair it by adding necessary codewords,
            # or just discard this perturbation.
            # For simplicity, discard.
            continue
        
        if len(current_sol) < best_size:
            best_size = len(current_sol)
            best_solution = current_sol
            
        current_time = time.time()
    
    # Convert indices back to codewords
    # Ensure distinctness (indices are distinct, so codewords are distinct)
    result = [all_words[i] for i in best_solution]
    
    # Final validation and redundancy removal just in case
    # (remove_redundant was called in the loop, but let's do one more pass on the final best)
    best_solution_indices = remove_redundant(best_solution)
    result = [all_words[i] for i in best_solution_indices]
    
    return result

# EVOLVE-BLOCK-END
