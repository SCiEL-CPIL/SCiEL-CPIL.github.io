import itertools
import time

def construct(n):
    start_time = time.time()
    
    # Helper for standard recursive construction
    def standard_construct(n):
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
                overlap = 0
                # Find max overlap
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n < 7:
        return standard_construct(n)
    
    # For n=7, try TSP-based ordering of blocks
    if n == 7:
        try:
            # 1. Generate permutations of 1..6
            perms = list(itertools.permutations(range(1, 7)))
            N = len(perms)
            
            # 2. Precompute overlap matrix
            # M[i][j] = overlap of perms[i] and perms[j]
            # Overlap is longest suffix of i that is prefix of j
            # Since length is 6, max overlap is 5 (if 6, they are same, but we don't repeat)
            # Actually if perms[i] == perms[j], overlap is 6. But i != j in the sequence.
            
            # Optimization: store perms as list of tuples
            # Use a flat list for M? Or list of lists?
            # List of lists is easier to index.
            M = [[0] * N for _ in range(N)]
            
            # Precompute
            # To speed up, we can unroll the loop or use the specific check
            for i in range(N):
                pi = perms[i]
                for j in range(N):
                    pj = perms[j]
                    # Check overlap
                    # k=5
                    if pi[1] == pj[0] and pi[2] == pj[1] and pi[3] == pj[2] and pi[4] == pj[3] and pi[5] == pj[4]:
                        M[i][j] = 5
                    elif pi[2] == pj[0] and pi[3] == pj[1] and pi[4] == pj[2] and pi[5] == pj[3]:
                        M[i][j] = 4
                    elif pi[3] == pj[0] and pi[4] == pj[1] and pi[5] == pj[2]:
                        M[i][j] = 3
                    elif pi[4] == pj[0] and pi[5] == pj[1]:
                        M[i][j] = 2
                    elif pi[5] == pj[0]:
                        M[i][j] = 1
                    else:
                        M[i][j] = 0
            
            elapsed = time.time() - start_time
            if elapsed > 60:
                return standard_construct(n)
            
            # 3. Greedy TSP
            # Start with 0
            visited = [False] * N
            visited[0] = True
            current = 0
            sequence = [0]
            
            for _ in range(N - 1):
                # Find next j not visited that maximizes M[current][j]
                max_overlap = -1
                best_j = -1
                row = M[current]
                for j in range(N):
                    if not visited[j]:
                        if row[j] > max_overlap:
                            max_overlap = row[j]
                            best_j = j
                            if max_overlap == 5: # Can't get better
                                break
                if best_j == -1:
                    # Should not happen
                    break
                visited[best_j] = True
                current = best_j
                sequence.append(best_j)
            
            elapsed = time.time() - start_time
            if elapsed > 100:
                return standard_construct(n)
            
            # 4. Construct word
            word = []
            # First block
            p0 = perms[sequence[0]]
            block0 = list(p0) + [7] + list(p0)
            word.extend(block0)
            
            for idx in range(1, N):
                pi = perms[sequence[idx-1]]
                pj = perms[sequence[idx]]
                # Calculate overlap again or store it?
                # We have M, but M[current][next] was used.
                # We need the overlap value to know how many chars to skip.
                # The overlap is M[sequence[idx-1]][sequence[idx]]
                ov = M[sequence[idx-1]][sequence[idx]]
                
                block = list(pj) + [7] + list(pj)
                # Append block[ov:]
                word.extend(block[ov:])
            
            return "".join(map(str, word))
            
        except Exception:
            return standard_construct(n)
    
    return standard_construct(n)
