import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Edge representation: (u, v) with u < v
    # We map edge index to (u, v) and vice versa
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
    
    # Faces: Each face is a set of 4 edges.
    # A face is defined by two dimensions i, j and a base vertex x with bits i, j = 0.
    # The 4 vertices are x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j).
    # The 4 edges are:
    # (x, x^(1<<i)), (x^(1<<j), x^(1<<i)^(1<<j))  [along i]
    # (x, x^(1<<j)), (x^(1<<i), x^(1<<i)^(1<<j))  [along j]
    
    faces = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all base vertices where bits i and j are 0
            # Mask for bits i and j
            mask = (1 << i) | (1 << j)
            for x in range(n_vertices):
                if (x & mask) == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e23 = edge_to_idx[(min(v2, v3), max(v2, v3))]
                    e02 = edge_to_idx[(min(v0, v2), max(v0, v2))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    
                    face_edges = (e01, e23, e02, e13)
                    face_idx = len(faces)
                    faces.append(face_edges)
                    
                    for e in face_edges:
                        edge_faces[e].append(face_idx)
    
    n_faces = len(faces)
    
    def run_restart(seed_val):
        random.seed(seed_val)
        # face_count[f] = number of edges in face f currently in the set
        face_count = [4] * n_faces  # Start with all edges
        in_set = [True] * n_edges
        
        # Phase 1: Greedy Deletion
        # We need to find faces with count 4.
        # Instead of scanning all faces, we can maintain a list of "bad" faces?
        # Or just scan. 672 faces is small.
        
        # To make it randomized, we can shuffle the order of edges we consider removing?
        # Better: While there are bad faces, pick one randomly and remove a random edge from it.
        
        # Optimization: Keep a list of bad faces? Updating it is expensive.
        # Just scan all faces to find bad ones.
        
        while True:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find a bad face
            bad_face_idx = -1
            # Randomize the search order to avoid bias
            # But scanning 672 items is fast.
            # To be truly random, we can pick a random face and check if it's bad?
            # If we pick a random face and it's not bad, we wasted time.
            # Probability of a face being bad decreases.
            # Let's just scan. It's O(672).
            
            # Actually, we can maintain a set of bad faces?
            # When we remove an edge, it affects 6 faces.
            # Let's try a simpler approach:
            # Shuffle edges. Iterate through edges? No, we need to remove edges from bad faces.
            
            # Let's stick to: Find any bad face, remove random edge from it.
            # To ensure we don't get stuck in a specific pattern, we randomize the face selection.
            
            # Efficiently finding a bad face:
            # We can just iterate.
            bad_found = False
            # Random start index for scanning faces
            start_idx = random.randint(0, n_faces - 1)
            for k in range(n_faces):
                f_idx = (start_idx + k) % n_faces
                if face_count[f_idx] == 4:
                    bad_face_idx = f_idx
                    bad_found = True
                    break
            
            if not bad_found:
                break
            
            # Pick a random edge from this face
            face_edges = faces[bad_face_idx]
            e_idx = face_edges[random.randint(0, 3)]
            
            # Remove edge e_idx
            in_set[e_idx] = False
            for f_idx in edge_faces[e_idx]:
                face_count[f_idx] -= 1
        
        # Phase 2: Add-back
        # Collect removed edges
        removed = [i for i in range(n_edges) if not in_set[i]]
        random.shuffle(removed)
        
        for e_idx in removed:
            if time.time() > deadline:
                break
            # Check if we can add e_idx
            can_add = True
            for f_idx in edge_faces[e_idx]:
                if face_count[f_idx] >= 3:
                    can_add = False
                    break
            
            if can_add:
                in_set[e_idx] = True
                for f_idx in edge_faces[e_idx]:
                    face_count[f_idx] += 1
        
        # Count edges
        count = sum(in_set)
        
        # Return list of edges
        result = []
        for i in range(n_edges):
            if in_set[i]:
                u, v = edges[i]
                result.append([u, v])
        
        return count, result

    best_count = 0
    best_result = []
    seed_val = 42
    
    while time.time() < deadline:
        count, result = run_restart(seed_val)
        if count > best_count:
            best_count = count
            best_result = result
            # print(f"New best: {best_count}") # Uncomment for debugging
        seed_val += 1
        # If we hit a very high number, maybe stop? 308 is upper bound.
        if best_count >= 308:
            break

    return best_result

# Keep the function name and signature as requested
# The prompt asks to keep the function names and signatures of the seed program.
# Seed: def construct():
# Return: list of edges [u, v]

# The code above defines construct().
# We need to make sure it's the only entry point or that construct is the one called.
# The prompt says "Keep the function names and signatures of the seed program."
# So we just output the code.
