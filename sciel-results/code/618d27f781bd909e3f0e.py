import time
import random
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy: Simulated Annealing with efficient conflict tracking.
    1. Precompute all faces (C4s) and edge-face incidence.
    2. Initialize with a dense random state or parity state.
    3. Repair initial state to remove any C4s.
    4. Run Simulated Annealing:
       - Move: Flip a random edge (add or remove).
       - Accept if it improves edge count (always).
       - Accept if it decreases edge count with probability exp(-delta/T).
       - This allows escaping local optima where greedy addition is stuck.
    5. Track best valid state found.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448
    
    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672
    
    # Precompute edge-to-faces mapping
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # State representation:
    # present: list of bools
    # face_counts: list of ints (0-4), number of present edges in each face.
    # A face is a conflict if face_counts[f] == 4.
    
    def repair(state):
        """Remove edges to eliminate all C4s. Returns updated state and face_counts."""
        face_counts = [0] * F
        for f_idx, face in enumerate(faces):
            count = 0
            for e in face:
                if state[e]:
                    count += 1
            face_counts[f_idx] = count
            
        # While there are conflicts, remove an edge from a random conflicting face
        # Heuristic: remove the edge that is in the most conflicting faces?
        # For speed, just pick random edge from a random conflicting face.
        # To be more effective, we can count how many conflicts each edge is in.
        
        # Let's do a slightly smarter repair:
        # Count conflicts per edge.
        while True:
            # Find any conflict
            conflict_face = -1
            for f_idx in range(F):
                if face_counts[f_idx] == 4:
                    conflict_face = f_idx
                    break
            
            if conflict_face == -1:
                break
            
            face = faces[conflict_face]
            # Pick an edge to remove.
            # To maximize future potential, maybe remove the one with highest degree in conflict graph?
            # Simple: random.
            e_to_remove = random.choice(face)
            
            state[e_to_remove] = False
            
            # Update face counts
            for f_idx in edge_faces[e_to_remove]:
                face_counts[f_idx] -= 1
                
        return state, face_counts

    def get_score(state):
        return sum(state)

    # Main Search Loop
    random.seed(42)
    
    best_score = -1
    best_state = None
    
    # We will run multiple restarts or one long annealing.
    # Given the time limit, one long annealing with restarts on stagnation might be best.
    # Or just multiple short annealings.
    
    # Let's try 50 restarts of annealing, each with a time budget.
    # Or one long run with cooling schedule.
    # Let's do one long run with adaptive restarts if stuck.
    
    # Initial state: Parity construction is a good dense start (224 edges).
    # Random 0.5 is ~224 edges.
    # Let's use parity.
    
    state = [False] * E
    for idx, (u, v) in enumerate(edge_list):
        if bin(u).count('1') % 2 == 0:
            state[idx] = True
            
    state, face_counts = repair(state)
    current_score = get_score(state)
    
    if current_score > best_score:
        best_score = current_score
        best_state = state[:]
        
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995  # Tune this based on time
    
    # We want to perform as many steps as possible.
    # Estimate steps:
    # Each step involves:
    # 1. Random choice of edge (O(1))
    # 2. Check if adding/removing creates conflict.
    #    If adding: check if any face containing e has count 3. (O(deg(e)) ~ 6)
    #    If removing: always valid (count decreases).
    # 3. Update face counts if accepted. (O(deg(e)) ~ 6)
    
    # Let's run until time limit.
    
    T = T_start
    steps = 0
    stagnation_counter = 0
    max_stagnation = 5000  # Restart if stuck for this many steps
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        e = random.randint(0, E - 1)
        
        if state[e]:
            # Try to remove edge e
            # This is always valid (reduces conflicts, doesn't create new ones)
            # Delta score = -1
            # Accept with probability 1 if T > 0? No, we want to maximize edges.
            # But removing might allow adding 2 others later.
            # In SA, we accept worse moves with prob exp(-delta/T).
            # delta = -1 (score decreases by 1).
            # prob = exp(1/T) > 1? No, exp(-(-1)/T) = exp(1/T).
            # Wait, standard SA: accept if delta > 0 or rand < exp(delta/T).
            # Here delta = new_score - old_score.
            # If we remove, delta = -1.
            # prob = exp(-1/T).
            
            # Actually, we should be careful. We want to maximize edges.
            # Removing an edge is a "bad" move in terms of score, but might be necessary to unblock others.
            
            # Let's calculate acceptance probability
            # If T is high, we accept more.
            # If T is low, we accept less.
            
            # However, we can optimize: if we remove an edge, we don't need to check conflicts.
            # We just update counts.
            
            # But we should only remove if it helps?
            # In standard SA, we accept based on probability.
            
            # Let's use a simpler heuristic for removal:
            # Only remove if it breaks a "potential" conflict?
            # No, let's stick to standard SA logic for robustness.
            
            # Check if this removal is part of a "useful" move?
            # For now, accept with probability exp(-1/T).
            
            if random.random() < (1.0 / T): # exp(-1/T) approx 1/T for small T? No.
                # Actually exp(-1/T).
                # Let's just use math.exp
                pass # Placeholder, will implement below
            
            # Wait, let's implement the check properly.
            
            # For removal:
            # New score = current_score - 1.
            # Accept if random() < exp((new_score - current_score) / T) = exp(-1/T).
            
            if random.random() < (1.0 / (T * 2.718)): # Rough approx, let's use math.exp
                pass
            
            # Let's use math.exp for correctness.
            
            import math
            p_accept = math.exp(-1.0 / T)
            
            if random.random() < p_accept:
                # Accept removal
                state[e] = False
                current_score -= 1
                for f_idx in edge_faces[e]:
                    face_counts[f_idx] -= 1
                stagnation_counter += 1
            else:
                stagnation_counter += 1 # Or not? Usually count all moves.
                
        else:
            # Try to add edge e
            # Check if adding creates a conflict
            # Conflict if any face containing e has count == 3
            can_add = True
            for f_idx in edge_faces[e]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                # Accept addition (always improves score)
                state[e] = True
                current_score += 1
                for f_idx in edge_faces[e]:
                    face_counts[f_idx] += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_state = state[:]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                # Cannot add without creating conflict
                # Reject
                stagnation_counter += 1
        
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Restart if stagnation
        if stagnation_counter > max_stagnation:
            # Restart with a new random state or perturb current
            # Perturb: remove 10 random edges, then repair
            for _ in range(10):
                e = random.randint(0, E - 1)
                if state[e]:
                    state[e] = False
                    for f_idx in edge_faces[e]:
                        face_counts[f_idx] -= 1
                    current_score -= 1
            
            # Repair
            # We need to recompute face_counts or continue with current.
            # Current face_counts are valid.
            # But we might have conflicts? No, we only removed edges.
            # So state is valid.
            # Just reset T and stagnation
            T = T_start
            stagnation_counter = 0
            
        steps += 1

    if best_state is None:
        return []
        
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
