import random
import time

def construct():
    n = 180
    triples_set = set()
    triples = []
    elem_to_triples = {i: [] for i in range(n + 1)}
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            den = b - a
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c > n:
                continue
            if c == b:
                continue
            tri = frozenset((a, b, c))
            if tri not in triples_set:
                triples_set.add(tri)
                idx = len(triples)
                triples.append(tri)
                for e in tri:
                    elem_to_triples[e].append(idx)
    
    num_triples = len(triples)
    
    # Precompute partners for fast delta calculation
    # partners[idx] is a dict: element -> tuple of other two
    partners = [{} for _ in range(num_triples)]
    for idx, tri in enumerate(triples):
        elems = list(tri)
        for i in range(3):
            e = elems[i]
            other1 = elems[(i+1)%3]
            other2 = elems[(i+2)%3]
            partners[idx][e] = (other1, other2)

    # State
    x = [1] * (n + 1)
    violated = [1] * num_triples # All violated initially
    violated_count = num_triples
    penalty = 200
    current_energy = penalty * violated_count - n
    best_x = x[:]
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    max_iters = 200000
    start_time = time.time()
    elements = list(range(1, n + 1))
    
    for it in range(max_iters):
        if time.time() - start_time > 55:
            break
        
        v = random.choice(elements)
        
        if x[v] == 0:
            # Add v
            delta_viol = 0
            for idx in elem_to_triples[v]:
                o1, o2 = partners[idx][v]
                if x[o1] == 1 and x[o2] == 1:
                    delta_viol += 1
            
            delta_energy = penalty * delta_viol - 1
            
            if delta_energy < 0 or random.random() < (2.718281828 ** (-delta_energy / T)):
                x[v] = 1
                current_energy += delta_energy
                violated_count += delta_viol
                
                for idx in elem_to_triples[v]:
                    o1, o2 = partners[idx][v]
                    if x[o1] == 1 and x[o2] == 1:
                        violated[idx] = 1
                    else:
                        violated[idx] = 0
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x[:]
        else:
            # Remove v
            delta_viol = 0
            for idx in elem_to_triples[v]:
                if violated[idx] == 1:
                    # It was violated, so removing v makes it un-violated
                    # No need to check others, they must be 1
                    delta_viol -= 1
            
            delta_energy = penalty * delta_viol + 1 # -(-1) = +1 to energy? 
            # E = P*V - S
            # Delta E = P * Delta V - Delta S
            # Delta V = -1 (if violated)
            # Delta S = -1
            # Delta E = P*(-1) - (-1) = -P + 1
            # So delta_energy = -penalty + 1 for each violated triple removed?
            # Wait, delta_viol is sum of changes.
            # If k triples become un-violated, delta_viol = -k.
            # Delta E = P * (-k) - (-1) = -Pk + 1.
            # My code: delta_viol is negative.
            # delta_energy = penalty * delta_viol - (-1) = penalty * delta_viol + 1.
            # This matches.
            
            if delta_energy < 0 or random.random() < (2.718281828 ** (-delta_energy / T)):
                x[v] = 0
                current_energy += delta_energy
                violated_count += delta_viol
                
                for idx in elem_to_triples[v]:
                    o1, o2 = partners[idx][v]
                    if x[o1] == 1 and x[o2] == 1:
                        violated[idx] = 1
                    else:
                        violated[idx] = 0
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    result = [i for i in range(1, n + 1) if best_x[i] == 1]
    if not result:
        result = [180]
    return result
