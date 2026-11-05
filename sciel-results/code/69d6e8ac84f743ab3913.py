import time

N = 40

def construct():
    t0 = time.time()
    
    # Build C_40
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Structural change: Use a two-hub construction to reduce chord count.
    # Hub 0: chords (0, k) for k in 2..10 (9 chords)
    # Hub 20: chords (20, k) for k in 22..30 (9 chords)  
    # This gives 18 chords total instead of 19.
    # 
    # Coverage analysis:
    # Chord (0, k) for k in 2..10 covers lengths k+1 and 41-k:
    #   k=2: 3, 39
    #   k=3: 4, 38
    #   k=4: 5, 37
    #   k=5: 6, 36
    #   k=6: 7, 35
    #   k=7: 8, 34
    #   k=8: 9, 33
    #   k=9: 10, 32
    #   k=10: 11, 31
    # 
    # Chord (20, k) for k in 22..30 (i.e., offsets 2..10 from 20):
    #   Chord (20, 22): arc 20->21->22->20 length 3, other arc length 38
    #   Chord (20, 23): arc length 4, other 37
    #   ...
    #   Chord (20, 30): arc length 11, other 30
    #
    # Combined coverage: lengths 3..11, 30..39 from both hubs.
    # Missing: lengths 12..29.
    # 
    # This doesn't work. Let me try a different approach.
    # 
    # Use chords that are "spread out" to cover more lengths.
    # Chord (i, j) covers lengths (j-i+1) and (40-(j-i)+1) = (41-(j-i))
    # if i < j.
    # 
    # Let's use a set of chords that covers all lengths 3..39 with 18 chords.
    # Each chord covers 2 lengths. 18 chords cover 36 lengths.
    # We need 37 lengths (3..39). So 18 chords can't cover all if each covers only 2.
    # Unless some chord covers more than 2 lengths via multi-chord cycles.
    
    # For now, use 19 chords but with a different distribution.
    # Chords: (0,2), (0,3), ..., (0,19), (1,21)
    # This is still 19 chords.
    
    # Let's just use the original 19-chord fan but verify correctness.
    for k in range(2, 21):
        edges.add((0, k))
        
    edge_list = list(edges)
    
    # Build cycles
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            cycles.append(list(range(N)))
        else:
            k1 = length - 1
            k2 = 41 - length
            
            found = False
            if 2 <= k1 <= 20 and (0, k1) in edges:
                cyc = list(range(k1 + 1))
                cycles.append(cyc)
                found = True
            elif 2 <= k2 <= 20 and (0, k2) in edges:
                cyc = [0] + list(range(39, k2 - 1, -1))
                cycles.append(cyc)
                found = True
            
            if not found:
                # Search for any chord forming this cycle length
                for (i, j) in edge_list:
                    if i > j:
                        i, j = j, i
                    arc1 = j - i + 1
                    arc2 = N - (j - i) + 1
                    if arc1 == length:
                        cyc = list(range(i, j + 1))
                        cycles.append(cyc)
                        found = True
                        break
                    elif arc2 == length:
                        cyc = list(range(j, N)) + list(range(0, i + 1))
                        cycles.append(cyc)
                        found = True
                        break
                if not found:
                    # Fallback: should not happen
                    k = length - 1
                    cyc = list(range(k + 1))
                    cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
