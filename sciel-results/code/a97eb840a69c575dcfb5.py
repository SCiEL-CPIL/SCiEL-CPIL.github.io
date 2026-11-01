import numpy as np
import time
import random
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s limit

    # Handle the specific case we know about
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)
    
    # Generic fallback
    return _generic_construct(q, n, radius, deadline, start_time)


def _solve_ternary_8_1(deadline, start_time):
    """
    Find a ternary covering code of length 8 and radius 1.
    
    Strategy: Use a bit-parallel representation with greedy set cover,
    then local search to minimize the number of codewords.
    """
    # Total space: 3^8 = 6561
    n_words = 3 ** 8
    
    # Precompute all words as tuples
    all_words = [tuple(w) for w in product(range(3), repeat=8)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres (Hamming ball of radius 1) for each word
    # Each sphere has 1 + 8*2 = 17 words
    sphere_indices = []
    for w in all_words:
        sphere = [word_to_idx[w]]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    sphere.append(word_to_idx[tuple(nw)])
        sphere_indices.append(sphere)
    
    # Greedy set cover with bit-parallel uncovered set
    # Represent uncovered as a numpy array of booleans
    uncovered = np.ones(n_words, dtype=np.bool_)
    code = []
    code_spheres = []
    
    while uncovered.any() and time.time() < deadline - 40:
        # Find the word that covers the most uncovered words
        # For speed, sample candidates
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) > 50:
            candidates = random.sample(list(uncovered_indices), 50)
        else:
            candidates = list(uncovered_indices)
        
        best_word = -1
        best_count = -1
        for wi in candidates:
            count = np.sum(uncovered[sphere_indices[wi]])
            if count > best_count:
                best_count = count
                best_word = wi
        
        if best_word == -1 or best_count <= 0:
            break
        
        code.append(all_words[best_word])
        code_spheres.append(sphere_indices[best_word])
        uncovered[sphere_indices[best_word]] = False
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if all words in its sphere are covered by other codewords
    for iteration in range(20000):
        if time.time() > deadline - 5:
            break
        
        # Compute coverage count for each word
        cc = np.zeros(n_words, dtype=np.int32)
        for sp in code_spheres:
            cc[sp] += 1
        
        removed_any = False
        # Try to remove codewords in random order
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(code):
                continue
            # Check if this codeword is redundant
            if cc[code_spheres[i]].min() > 1:
                # Remove it
                for j in code_spheres[i]:
                    cc[j] -= 1
                code.pop(i)
                code_spheres.pop(i)
                removed_any = True
                # Adjust indices
                if i < len(indices):
                    indices = list(range(len(code)))
                    random.shuffle(indices)
        
        if not removed_any:
            # Try swaps: remove one codeword, add another that covers more
            cc = np.zeros(n_words, dtype=np.int32)
            for sp in code_spheres:
                cc[sp] += 1
            
            found_improvement = False
            for i in range(len(code)):
                if cc[code_spheres[i]].min() <= 1:
                    continue  # Can't remove this one without losing coverage
                
                # Try removing and adding a different codeword
                # Temporarily remove
                cc[code_spheres[i]] -= 1
                # Find words not covered
                uncovered_mask = cc == 0
                if not uncovered_mask.any():
                    # This codeword was redundant, remove it
                    code.pop(i)
                    code_spheres.pop(i)
                    found_improvement = True
                    break
                else:
                    # Try to find a codeword that covers many uncovered words
                    uncovered_indices = np.where(uncovered_mask)[0]
                    best_new = -1
                    best_count = -1
                    for wi in random.sample(list(uncovered_indices), min(20, len(uncovered_indices))):
                        count = np.sum(uncovered_mask[sphere_indices[wi]])
                        if count > best_count:
                            best_count = count
                            best_new = wi
                    
                    if best_new != -1 and best_count > 0:
                        # Add the new codeword
                        code.append(all_words[best_new])
                        code_spheres.append(sphere_indices[best_new])
                        cc[sphere_indices[best_new]] += 1
                        found_improvement = True
                        break
                    else:
                        # Restore
                        cc[code_spheres[i]] += 1
            
            if not found_improvement:
                break
    
    return [list(cw) for cw in code]


def _generic_construct(q, n, radius, deadline, start_time):
    """Generic construction using greedy + local search."""
    total_words = q ** n
    if total_words > 100000:
        # Too large for full enumeration
        return _simple_greedy(q, n, radius, deadline)
    
    # Generate all words
    all_words = [tuple(w) for w in product(range(q), repeat=n)]
    n_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres
    sphere_indices = []
    for w in all_words:
        sphere = [word_to_idx[w]]
        for d in range(1, radius + 1):
            for pos in product(range(n), repeat=d):
                if len(set(pos)) < d:
                    continue
                for vals in product(range(q), repeat=d):
                    nw = list(w)
                    for p, v in zip(pos, vals):
                        if nw[p] != v:
                            nw[p] = v
                            sphere.append(word_to_idx[tuple(nw)])
                            break
        # Remove duplicates
        sphere_indices.append(list(set(sphere)))
    
    # Greedy set cover
    uncovered = np.ones(n_words, dtype=np.bool_)
    code = []
    code_spheres = []
    
    while uncovered.any() and time.time() < deadline - 20:
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) > 50:
            candidates = random.sample(list(uncovered_indices), 50)
        else:
            candidates = list(uncovered_indices)
        
        best_word = -1
        best_count = -1
        for wi in candidates:
            count = np.sum(uncovered[sphere_indices[wi]])
            if count > best_count:
                best_count = count
                best_word = wi
        
        if best_word == -1 or best_count <= 0:
            break
        
        code.append(all_words[best_word])
        code_spheres.append(sphere_indices[best_word])
        uncovered[sphere_indices[best_word]] = False
    
    # Local search
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        cc = np.zeros(n_words, dtype=np.int32)
        for sp in code_spheres:
            cc[sp] += 1
        
        removed_any = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(code):
                continue
            if cc[code_spheres[i]].min() > 1:
                for j in code_spheres[i]:
                    cc[j] -= 1
                code.pop(i)
                code_spheres.pop(i)
                removed_any = True
                if i < len(indices):
                    indices = list(range(len(code)))
                    random.shuffle(indices)
        
        if not removed_any:
            break
    
    return [list(cw) for cw in code]


def _simple_greedy(q, n, radius, deadline):
    """Simple greedy for large instances - returns empty list (invalid)."""
    return []
