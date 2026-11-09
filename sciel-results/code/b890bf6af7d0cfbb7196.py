import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # We aim for n=34. If we find it, we return it.
    # If not, we might try smaller n or return best effort.
    # Given the difficulty, finding 34 is the main goal.
    
    n = 34
    triples = list(combinations(range(n), 3))
    n_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    quads = list(combinations(range(n), 4))
    n_quads = len(quads)
    quad_triples = []
    for q in quads:
        ts = [triple_to_idx[(q[0], q[1], q[2])], 
              triple_to_idx[(q[0], q[1], q[3])], 
              triple_to_idx[(q[0], q[2], q[3])], 
              triple_to_idx[(q[1], q[2], q[3])]]
        quad_triples.append(ts)
        
    # Precompute 5-sets and their triples
    quints = list(combinations(range(n), 5))
    n_quints = len(quints)
    quint_triples = []
    for q in quints:
        ts = [triple_to_idx[(q[i], q[j], q[k])] for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
        quint_triples.append(ts)
        
    # For each triple, list the quads and quints it belongs to
    triple_to_quads = [[] for _ in range(n_triples)]
    for i, ts in enumerate(quad_triples):
        for t in ts:
            triple_to_quads[t].append(i)
            
    triple_to_quints = [[] for _ in range(n_triples)]
    for i, ts in enumerate(quint_triples):
        for t in ts:
            triple_to_quints[t].append(i)
            
    # Initialize random coloring
    colors = np.random.randint(0, 2, size=n_triples)
    
    # Initialize violation counts
    viol4 = np.zeros(n_quads, dtype=np.int32)
    viol5 = np.zeros(n_quints, dtype=np.int32)
    
    # Calculate initial violations
    for i, ts in enumerate(quad_triples):
        viol4[i] = np.sum(1 - colors[ts]) # Red is 0, so 1-0=1
        # Wait, colors are 0 (Red) or 1 (Blue).
        # Red count = sum(1 - colors)
        # Blue count = sum(colors)
        
    for i, ts in enumerate(quint_triples):
        viol5[i] = np.sum(colors[ts])
        
    # Sets of violated indices
    viol4_set = set(np.where(viol4 == 4)[0])
    viol5_set = set(np.where(viol5 == 10)[0])
    
    best_n = 0
    best_colors = None
    
    while time.time() < deadline:
        if not viol4_set and not viol5_set:
            best_n = n
            best_colors = colors.copy()
            break
            
        # Pick a violation
        if viol4_set:
            if viol5_set and random.random() < 0.5:
                q_idx = random.choice(list(viol5_set))
                ts = quint_triples[q_idx]
            else:
                q_idx = random.choice(list(viol4_set))
                ts = quad_triples[q_idx]
        else:
            q_idx = random.choice(list(viol5_set))
            ts = quint_triples[q_idx]
            
        # Pick a random triple from the violating set
        t_idx = random.choice(ts)
        
        # Flip color
        old_color = colors[t_idx]
        new_color = 1 - old_color
        colors[t_idx] = new_color
        
        # Update violations for quads containing t_idx
        for q_idx in triple_to_quads[t_idx]:
            if old_color == 0: # Was Red, now Blue
                viol4[q_idx] -= 1
                if viol4[q_idx] == 3: # No longer violated (was 4)
                    viol4_set.discard(q_idx)
                # If it becomes 4? No, it decreased.
            else: # Was Blue, now Red
                viol4[q_idx] += 1
                if viol4[q_idx] == 4: # Now violated
                    viol4_set.add(q_idx)
                
        # Update violations for quints containing t_idx
        for q_idx in triple_to_quints[t_idx]:
            if old_color == 0: # Was Red, now Blue
                viol5[q_idx] += 1
                if viol5[q_idx] == 10: # Now violated
                    viol5_set.add(q_idx)
            else: # Was Blue, now Red
                viol5[q_idx] -= 1
                if viol5[q_idx] == 9: # No longer violated (was 10)
                    viol5_set.discard(q_idx)

    if best_n > 0:
        return {'n': best_n, 'colors': best_colors.tolist()}
    
    # If we didn't find a solution for 34, we might try to salvage a smaller one?
    # Or just return a valid coloring for a smaller n if we had one?
    # We didn't track smaller n solutions.
    # We can try n=33 quickly if time permits.
    
    if time.time() < deadline - 5:
         # Try n=33
         n = 33
         # ... repeat search ...
         # To save space, I'll just return the current best or a fallback.
         # Actually, if we fail 34, returning 9 is bad.
         # Let's implement a quick search for n=33 if 34 fails.
         
         # Re-initialize for n=33
         triples = list(combinations(range(n), 3))
         n_triples = len(triples)
         triple_to_idx = {t: i for i, t in enumerate(triples)}
         quads = list(combinations(range(n), 4))
         n_quads = len(quads)
         quad_triples = []
         for q in quads:
             ts = [triple_to_idx[(q[0], q[1], q[2])], 
                   triple_to_idx[(q[0], q[1], q[3])], 
                   triple_to_idx[(q[0], q[2], q[3])], 
                   triple_to_idx[(q[1], q[2], q[3])]]
             quad_triples.append(ts)
         quints = list(combinations(range(n), 5))
         n_quints = len(quints)
         quint_triples = []
         for q in quints:
             ts = [triple_to_idx[(q[i], q[j], q[k])] for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
             quint_triples.append(ts)
         triple_to_quads = [[] for _ in range(n_triples)]
         for i, ts in enumerate(quad_triples):
             for t in ts:
                 triple_to_quads[t].append(i)
         triple_to_quints = [[] for _ in range(n_triples)]
         for i, ts in enumerate(quint_triples):
             for t in ts:
                 triple_to_quints[t].append(i)
                 
         colors = np.random.randint(0, 2, size=n_triples)
         viol4 = np.zeros(n_quads, dtype=np.int32)
         viol5 = np.zeros(n_quints, dtype=np.int32)
         
         for i, ts in enumerate(quad_triples):
             viol4[i] = np.sum(1 - colors[ts])
         for i, ts in enumerate(quint_triples):
             viol5[i] = np.sum(colors[ts])
             
         viol4_set = set(np.where(viol4 == 4)[0])
         viol5_set = set(np.where(viol5 == 10)[0])
         
         while time.time() < deadline:
             if not viol4_set and not viol5_set:
                 best_n = n
                 best_colors = colors.copy()
                 break
                 
             if viol4_set:
                 if viol5_set and random.random() < 0.5:
                     q_idx = random.choice(list(viol5_set))
                     ts = quint_triples[q_idx]
                 else:
                     q_idx = random.choice(list(viol4_set))
                     ts = quad_triples[q_idx]
             else:
                 q_idx = random.choice(list(viol5_set))
                 ts = quint_triples[q_idx]
                 
             t_idx = random.choice(ts)
             old_color = colors[t_idx]
             new_color = 1 - old_color
             colors[t_idx] = new_color
             
             for q_idx in triple_to_quads[t_idx]:
                 if old_color == 0:
                     viol4[q_idx] -= 1
                     if viol4[q_idx] == 3:
                         viol4_set.discard(q_idx)
                 else:
                     viol4[q_idx] += 1
                     if viol4[q_idx] == 4:
                         viol4_set.add(q_idx)
                         
             for q_idx in triple_to_quints[t_idx]:
                 if old_color == 0:
                     viol5[q_idx] += 1
                     if viol5[q_idx] == 10:
                         viol5_set.add(q_idx)
                 else:
                     viol5[q_idx] -= 1
                     if viol5[q_idx] == 9:
                         viol5_set.discard(q_idx)

    if best_n > 0:
        return {'n': best_n, 'colors': best_colors.tolist()}
        
    # Fallback to seed
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
