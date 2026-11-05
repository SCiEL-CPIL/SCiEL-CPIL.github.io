import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 180
    
    # Precompute P_a
    P_a = {a: [] for a in range(1, 91)}
    for a in range(1, 91):
        a_sq = a * a
        for d in range(1, a_sq + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b > c:
                    b, c = c, b
                if b != c and 91 <= b <= 180 and 91 <= c <= 180:
                    P_a[a].append((b, c))
        P_a[a] = list(set(P_a[a]))
        
    # Precompute C_ab
    C_ab = {}
    for a in range(1, 91):
        for b in range(a + 1, 91):
            c_val = (a * b) / (b - a)
            if c_val == int(c_val):
                c = int(c_val)
                if 91 <= c <= 180:
                    if (a, b) not in C_ab:
                        C_ab[(a, b)] = []
                    C_ab[(a, b)].append(c)
                    
    def get_mis_size_and_set(vertices, edge_set, rng, num_trials=5):
        if not vertices:
            return 0, set()
        
        best_size = 0
        best_set = set()
        v_set = set(vertices)
        
        for _ in range(num_trials):
            selected = []
            remaining = set(vertices)
            v_list = list(remaining)
            rng.shuffle(v_list)
            
            for v in v_list:
                if v not in remaining:
                    continue
                conflict = False
                for u in selected:
                    if (min(u, v), max(u, v)) in edge_set:
                        conflict = True
                        break
                if not conflict:
                    selected.append(v)
                    remaining.remove(v)
                    for u in remaining:
                        if (min(u, v), max(u, v)) in edge_set:
                            remaining.discard(u)
                            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
                
        return best_size, best_set

    def evaluate_A(A, rng, return_set=False):
        v_forb = set()
        for a in A:
            for b in A:
                if a < b:
                    if (a, b) in C_ab:
                        v_forb.update(C_ab[(a, b)])
        
        available_B = [x for x in range(91, 181) if x not in v_forb]
        
        edges = []
        for a in A:
            for pair in P_a[a]:
                u, v = pair
                if u in available_B and v in available_B:
                    edges.append((min(u, v), max(u, v)))
        
        edges = list(set(edges))
        edge_set = set(edges)
        
        size, mis_set = get_mis_size_and_set(available_B, edge_set, rng, num_trials=3)
        
        if return_set:
            return size, mis_set
        return size

    best_score = 0
    best_A = set()
    best_B_set = set()
    
    rng = random.Random(42)
    
    # Initial evaluation
    best_B_size, best_B_set = evaluate_A(best_A, rng, return_set=True)
    best_score = len(best_A) + best_B_size
    
    A = set(best_A)
    
    all_candidates = list(range(1, 91))
    
    for step in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        # Add
        if len(A) < 90:
            a = rng.choice([x for x in all_candidates if x not in A])
            new_A = A | {a}
            new_B_size = evaluate_A(new_A, rng)
            new_score = len(new_A) + new_B_size
            
            if new_score > best_score:
                A = new_A
                best_A = A
                best_B_size = new_B_size
                best_score = new_score
                # Update best_B_set later or now?
                # To save time, only compute set if we update best
                _, best_B_set = evaluate_A(best_A, rng, return_set=True)
            elif new_score == best_score and rng.random() < 0.05:
                A = new_A
                # Do not update best_A/best_score if equal, just move A
                
        # Remove
        if len(A) > 0:
            a = rng.choice(list(A))
            new_A = A - {a}
            new_B_size = evaluate_A(new_A, rng)
            new_score = len(new_A) + new_B_size
            
            if new_score > best_score:
                A = new_A
                best_A = A
                best_B_size = new_B_size
                best_score = new_score
                _, best_B_set = evaluate_A(best_A, rng, return_set=True)
            elif new_score == best_score and rng.random() < 0.05:
                A = new_A

    # Final polish: try to improve B_set for best_A
    # Re-evaluate best_A with more trials
    v_forb = set()
    for a in best_A:
        for b in best_A:
            if a < b:
                if (a, b) in C_ab:
                    v_forb.update(C_ab[(a, b)])
    available_B = [x for x in range(91, 181) if x not in v_forb]
    edges = []
    for a in best_A:
        for pair in P_a[a]:
            u, v = pair
            if u in available_B and v in available_B:
                edges.append((min(u, v), max(u, v)))
    edges = list(set(edges))
    edge_set = set(edges)
    
    _, best_B_set = get_mis_size_and_set(available_B, edge_set, rng, num_trials=50)
    
    result = sorted(list(best_A) + list(best_B_set))
    
    if not result:
        return list(range(91, 181))
        
    return result

# EVOLVE-BLOCK-END
