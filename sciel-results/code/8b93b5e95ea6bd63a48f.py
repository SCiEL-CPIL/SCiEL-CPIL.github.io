def construct(n):
        word = [1]
        for m in range(2, n + 1):
            # Extract perms of length m-1
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            if len(perms) <= 1:
                # Trivial case
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    if not new:
                        new = block
                    else:
                        # Find overlap
                        overlap = 0
                        max_ov = min(len(new), len(block))
                        for o in range(max_ov, -1, -1):
                            if new[len(new) - o:] == block[:o]:
                                overlap = o
                                break
                        new += block[overlap:]
                word = new
                continue

            # Greedy Nearest Neighbor to order perms
            # Cost is 1 - overlap
            # We want to minimize total length, so maximize total overlap.
            
            import random
            # To be deterministic, we don't use random, but we can try multiple starts?
            # For now, start with the first one.
            
            unvisited = set(perms)
            unvisited.remove(perms[0])
            current = perms[0]
            order = [current]
            
            while unvisited:
                best_p = None
                best_overlap = -1
                for p in unvisited:
                    # Calculate overlap between current and p
                    # current is tuple, p is tuple
                    # Max overlap is m-1
                    # But they are distinct, so max overlap is m-2?
                    # No, they can share m-1 if they are the same, but they are distinct.
                    # So max overlap is m-2?
                    # Example: 12 and 23. Overlap 2 (2).
                    # 12 and 13. Overlap 1 (1).
                    # 12 and 31. Overlap 1 (1, 2 vs 3, 1? No. 12 suffix 2, 31 prefix 3. No. Suffix 1, prefix 3. No. Overlap 0).
                    # Wait, 12 and 21. Suffix 2, prefix 2. Overlap 1.
                    
                    ov = 0
                    max_ov = m - 2 # Since distinct, max shared is m-2?
                    # Actually, if m=3, perms are len 2.
                    # 12 and 23. Suffix 2, prefix 2. Overlap 1.
                    # 12 and 13. Suffix 2, prefix 1. No. Suffix 1, prefix 1. Yes. Overlap 1.
                    # So max overlap is m-2?
                    # If m=4, perms len 3.
                    # 123 and 234. Suffix 23, prefix 23. Overlap 2.
                    # 123 and 124. Suffix 23, prefix 12. No. Suffix 3, prefix 1. No. Suffix 2, prefix 1. No. Suffix 1, prefix 1. Yes. Overlap 1.
                    # So yes, max overlap is m-2.
                    
                    for k in range(m - 2, -1, -1):
                        if current[-k:] == p[:k]:
                            ov = k
                            break
                    if ov > best_overlap:
                        best_overlap = ov
                        best_p = p
                
                order.append(best_p)
                unvisited.remove(best_p)
                current = best_p
            
            # Construct new word
            new = []
            for p in order:
                block = list(p) + [m] + list(p)
                if not new:
                    new = block
                else:
                    overlap = 0
                    max_ov = min(len(new), len(block))
                    for o in range(max_ov, -1, -1):
                        if new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
            word = new
        
        return "".join(map(str, word))
