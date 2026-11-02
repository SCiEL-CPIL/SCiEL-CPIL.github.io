import itertools
import time

def construct(q, n, radius):
    # Fallback: Tetracode direct-sum construction
    def construct_tetracode(q, n, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n <= 4:
            return [h[:n] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Direct-sum construction: 9 * 3^2 = 81 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    # Verify coverage
    all_space = list(itertools.product(range(3), repeat=6))
    covered = set()
    for c in code:
        c = tuple(c)
        covered.add(c)
        for i in range(6):
            for d in [-1, 1]:
                nc = list(c)
                nc[i] = (nc[i] + d) % 3
                covered.add(tuple(nc))
    
    if len(covered) != 729:
        return code  # Should always be covered by construction

    # Greedy removal: try to remove codewords while maintaining coverage
    # Use efficient bitmask representation
    word_to_idx = {w: i for i, w in enumerate(all_space)}
    
    # Precompute for each codeword which words it covers
    def get_covered_words(c):
        c = tuple(c)
        words = {c}
        for i in range(6):
            for d in [-1, 1]:
                nc = list(c)
                nc[i] = (nc[i] + d) % 3
                words.add(tuple(nc))
        return words
    
    # For efficiency, represent coverage as a boolean array
    cover_count = [0] * 729
    for c in code:
        for w in get_covered_words(c):
            cover_count[word_to_idx[w]] += 1
    
    improved_code = list(code)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 2.0:
            break
        for i in range(len(improved_code)):
            if time.time() > deadline - 1.0:
                break
            c = improved_code[i]
            # Check if removing c leaves all words covered
            can_remove = True
            for w in get_covered_words(c):
                if cover_count[word_to_idx[w]] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove c and update cover counts
                for w in get_covered_words(c):
                    cover_count[word_to_idx[w]] -= 1
                improved_code.pop(i)
                changed = True
                break
    
    # Now try greedy addition: find words whose addition allows removing multiple codewords
    # This is more complex, so we'll do a simpler local search
    # Try swapping: for each pair of codewords, see if replacing one with another helps
    
    # Alternative approach: use the structure. The direct sum has a product structure.
    # C5 (length 5, radius 1) combined with C1 (length 1, radius 0) gives C6.
    # If we can find a smaller C5, we get a smaller C6.
    
    # Let's try to find a covering code for n=5, q=3, r=1
    # Space size: 243, sphere size: 1 + 5*2 = 11
    # Lower bound: ceil(243/11) = 23
    # Known: K_3(5,1) = 27 (I think)
    
    # Greedy for n=5
    if time.time() < deadline - 30.0:
        all_words_5 = list(itertools.product(range(3), repeat=5))
        word_to_idx_5 = {w: i for i, w in enumerate(all_words_5)}
        
        # Precompute spheres for n=5
        spheres_5 = []
        for w in all_words_5:
            sphere = [w]
            for i in range(5):
                for d in [-1, 1]:
                    nw = list(w)
                    nw[i] = (nw[i] + d) % 3
                    sphere.append(tuple(nw))
            spheres_5.append(sphere)
        
        covered_5 = [False] * 243
        code_5 = []
        
        while True:
            if time.time() > deadline - 10.0:
                break
            best_idx = -1
            best_count = 0
            
            for i, w in enumerate(all_words_5):
                count = sum(1 for v in spheres_5[i] if not covered_5[word_to_idx_5[v]])
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_count == 0:
                break
            
            code_5.append(all_words_5[best_idx])
            for v in spheres_5[best_idx]:
                covered_5[word_to_idx_5[v]] = True
        
        if all(covered_5):
            # Construct final code from C5
            final_code = []
            for c in code_5:
                for x in range(3):
                    final_code.append(list(c) + [x])
            
            if len(final_code) < len(improved_code):
                improved_code = final_code
    
    # Try to further improve by local search
    if time.time() < deadline - 5.0:
        # For each codeword, try replacing it with a different word
        # and see if we can then remove more codewords
        for _ in range(3):  # A few rounds
            if time.time() > deadline - 3.0:
                break
            
            # Recompute cover counts
            cover_count = [0] * 729
            for c in improved_code:
                for w in get_covered_words(c):
                    cover_count[word_to_idx[w]] += 1
            
            made_improvement = False
            for i in range(len(improved_code)):
                if time.time() > deadline - 2.0:
                    break
                c = improved_code[i]
                # Try each possible replacement
                for new_word in all_space:
                    new_word = list(new_word)
                    if new_word == c:
                        continue
                    # Check if adding new_word and removing c still covers everything
                    # Words only covered by c must be covered by new_word
                    can_replace = True
                    for w in get_covered_words(tuple(c)):
                        if cover_count[word_to_idx[w]] == 1:
                            # w is only covered by c, so new_word must cover w
                            if tuple(w) not in get_covered_words(tuple(new_word)):
                                can_replace = False
                                break
                    if can_replace:
                        # Do the replacement
                        for w in get_covered_words(tuple(c)):
                            cover_count[word_to_idx[w]] -= 1
                        for w in get_covered_words(tuple(new_word)):
                            cover_count[word_to_idx[w]] += 1
                        improved_code[i] = new_word
                        made_improvement = True
                        break
            
            if not made_improvement:
                break
            
            # Try removing again
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 2.0:
                    break
                for i in range(len(improved_code)):
                    if time.time() > deadline - 1.0:
                        break
                    c = improved_code[i]
                    can_remove = True
                    for w in get_covered_words(tuple(c)):
                        if cover_count[word_to_idx[w]] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for w in get_covered_words(tuple(c)):
                            cover_count[word_to_idx[w]] -= 1
                        improved_code.pop(i)
                        changed = True
                        break
    
    return improved_code
