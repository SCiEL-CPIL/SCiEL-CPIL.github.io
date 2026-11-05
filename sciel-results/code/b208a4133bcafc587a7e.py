"""Combined: MCS-based ordering + face-count local search with spectral initialization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute faces (4-cycles)
    faces = []
    edge_faces = [[] for _ in range(m)]
    
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v3, v4), max(v3, v4))
                    e3 = (min(v1, v3), max(v1, v3))
                    e4 = (min(v2, v4), max(v2, v4))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                    
                    face_edges = (idx1, idx2, idx3, idx4)
                    faces.append(face_edges)
                    
                    for e_idx in face_edges:
                        edge_faces[e_idx].append(len(faces) - 1)
    
    num_faces = len(faces)
    
    # Build conflict graph: two edges conflict if they are in the same face
    conflict_adj = [[] for _ in range(m)]
    for f_idx in range(num_faces):
        fe = faces[f_idx]
        for a in range(4):
            for b in range(a+1, 4):
                if fe[a] not in conflict_adj[fe[b]]:
                    conflict_adj[fe[a]].append(fe[b])
                    conflict_adj[fe[b]].append(fe[a])
    
    def greedy_polish(chosen, face_counts):
        """Add all possible edges."""
        improved = True
        count = int(chosen.sum())
        while improved:
            improved = False
            for idx in range(m):
                if not chosen[idx]:
                    can_add = True
                    for f_idx in edge_faces[idx]:
                        if face_counts[f_idx] >= 3:
                            can_add = False
                            break
                    if can_add:
                        chosen[idx] = True
                        count += 1
                        for f_idx in edge_faces[idx]:
                            face_counts[f_idx] += 1
                        improved = True
        return count
    
    def walk_sat_local_search(chosen, face_counts, count, max_time=None):
        """WalkSAT-style local search for C4-free edge selection.
        
        The objective is to maximize the number of selected edges while ensuring
        no face has all 4 edges selected. We use a hybrid approach:
        - Primary moves: add edges that don't violate constraints (greedy)
        - Secondary moves: swap operations to escape local optima
        - Tertiary moves: random flips with probability based on constraint violation
        
        We track:
        - face_counts[f]: number of selected edges in face f (0-4)
        - A face is "satisfied" if face_counts[f] <= 3
        - A face is "violated" if face_counts[f] == 4
        """
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
        no_improve = 0
        max_no_improve = 50000
        
        # Track violated faces for quick access
        violated_faces = set()
        for f_idx in range(num_faces):
            if face_counts[f_idx] == 4:
                violated_faces.add(f_idx)
        
        # WalkSAT parameters
        p_add = 0.3      # probability of trying to add an edge
        p_swap = 0.4     # probability of trying a swap
        p_flip = 0.3     # probability of random flip
        flip_threshold = 0.7  # threshold for accepting worsening moves
        
        T = 1.0
        cooling = 0.99995
        T_min = 0.01
        
        while time.time() < max_time:
            # Choose move type
            r = random.random()
            
            if r < p_add:
                # Try to add an edge
                # Prefer edges that are in fewer violated faces
                unselected = [i for i in range(m) if not chosen[i]]
                if not unselected:
                    continue
                
                # Score each unselected edge by how many violated faces it's in
                # (we want to add edges that don't create new violations)
                scored = []
                for idx in unselected:
                    score = 0
                    for f_idx in edge_faces[idx]:
                        if face_counts[f_idx] >= 3:
                            score += 1  # penalty for being in near-full/full faces
                    scored.append((score, idx))
                
                # Pick from the best (lowest score) candidates
                scored.sort()
                # Take top k candidates and pick randomly
                k = min(10, len(scored))
                candidates = [s[1] for s in scored[:k]]
                idx = random.choice(candidates)
                
                # Check if we can add
                can_add = True
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] >= 3:
                        can_add = False
                        break
                
                if can_add:
                    chosen[idx] = True
                    count += 1
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] += 1
                        if face_counts[f_idx] == 4:
                            violated_faces.add(f_idx)
                    
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        best_face_counts = face_counts.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif r < p_add + p_swap:
                # Try a swap: remove one edge, add another
                # Find an edge in a violated or near-violated face to remove
                potential_removals = []
                for f_idx in violated_faces:
                    if face_counts[f_idx] == 4:
                        for e in faces[f_idx]:
                            if chosen[e]:
                                potential_removals.append(e)
                
                # Also consider edges in faces with count 3 (to make room)
                for f_idx in range(num_faces):
                    if face_counts[f_idx] == 3:
                        for e in faces[f_idx]:
                            if chosen[e]:
                                potential_removals.append(e)
                
                if not potential_removals:
                    # No good removals, try random
                    selected = [i for i in range(m) if chosen[i]]
                    if not selected:
                        continue
                    rem_idx = random.choice(selected)
                else:
                    rem_idx = random.choice(potential_removals[:20])
                
                # Try to find an edge to add after removing rem_idx
                # Tentatively remove
                for f_idx in edge_faces[rem_idx]:
                    face_counts[f_idx] -= 1
                    if face_counts[f_idx] == 3 and f_idx in violated_faces:
                        violated_faces.discard(f_idx)
                chosen[rem_idx] = False
                count -= 1
                
                # Now try to add edges
                added = False
                unselected = [i for i in range(m) if not chosen[i]]
                random.shuffle(unselected)
                for add_idx in unselected[:50]:  # Try up to 50 candidates
                    can_add = True
                    for f_idx in edge_faces[add_idx]:
                        if face_counts[f_idx] >= 3:
                            can_add = False
                            break
                    if can_add:
                        chosen[add_idx] = True
                        count += 1
                        for f_idx in edge_faces[add_idx]:
                            face_counts[f_idx] += 1
                            if face_counts[f_idx] == 4:
                                violated_faces.add(f_idx)
                        added = True
                        break
                
                if not added:
                    # Revert
                    chosen[rem_idx] = True
                    count += 1
                    for f_idx in edge_faces[rem_idx]:
                        face_counts[f_idx] += 1
                        if face_counts[f_idx] == 4:
                            violated_faces.add(f_idx)
                
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    best_face_counts = face_counts.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
            else:
                # Random flip with acceptance probability
                idx = random.randint(0, m - 1)
                
                if chosen[idx]:
                    # Remove it
                    old_count = count
                    chosen[idx] = False
                    count -= 1
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] -= 1
                        if face_counts[f_idx] == 3 and f_idx in violated_faces:
                            violated_faces.discard(f_idx)
                    
                    # Accept if it improves or with probability based on T
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        best_face_counts = face_counts.copy()
                        no_improve = 0
                    else:
                        # Accept with some probability to escape local optima
                        if random.random() < 0.3:
                            no_improve += 1
                        else:
                            # Revert
                            chosen[idx] = True
                            count = old_count
                            for f_idx in edge_faces[idx]:
                                face_counts[f_idx] += 1
                                if face_counts[f_idx] == 4:
                                    violated_faces.add(f_idx)
                            no_improve += 1
                else:
                    # Add it if possible
                    can_add = True
                    for f_idx in edge_faces[idx]:
                        if face_counts[f_idx] >= 3:
                            can_add = False
                            break
                    
                    if can_add:
                        chosen[idx] = True
                        count += 1
                        for f_idx in edge_faces[idx]:
                            face_counts[f_idx] += 1
                            if face_counts[f_idx] == 4:
                                violated_faces.add(f_idx)
                        
                        if count > best_count:
                            best_count = count
                            best_chosen = chosen.copy()
                            best_face_counts = face_counts.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                chosen = best_chosen.copy()
                face_counts = best_face_counts.copy()
                count = best_count
                no_improve = 0
                # Rebuild violated_faces
                violated_faces = set()
                for f_idx in range(num_faces):
                    if face_counts[f_idx] == 4:
                        violated_faces.add(f_idx)
        
        return best_chosen, best_face_counts, best_count
    
    def mcs_ordering():
        """Maximum Cardinality Search to get an ordering of edges.
        Returns a list of edge indices in the order they were 'discovered'."""
        chosen = np.zeros(m, dtype=bool)
        face_counts = np.zeros(num_faces, dtype=np.int32)
        remaining = set(range(m))
        
        order = []
        while remaining:
            best_idx = -1
            best_score = -1
            for idx in remaining:
                score = 0
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] < 3:
                        score += 1
                if score > best_score or (score == best_score and (best_idx == -1 or random.random() < 0.5)):
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            order.append(best_idx)
            remaining.discard(best_idx)
            
            can_add = True
            for f_idx in edge_faces[best_idx]:
                if face_counts[f_idx] >= 3:
                    can_add = False
                    break
            
            if can_add:
                chosen[best_idx] = True
                for f_idx in edge_faces[best_idx]:
                    face_counts[f_idx] += 1
        
        return chosen, face_counts, int(chosen.sum())
    
    def spectral_init():
        """Use Fiedler vector of conflict graph for initial selection."""
        row = []
        col = []
        data = []
        for i in range(m):
            for j in conflict_adj[i]:
                if i < j:
                    row.append(i)
                    col.append(j)
                    data.append(1.0)
                    row.append(j)
                    col.append(i)
                    data.append(1.0)
        
        if len(data) == 0:
            return np.random.randn(m)
        
        A = csr_matrix((data, (row, col)), shape=(m, m))
        L = laplacian(A, normed=True)
        
        try:
            eigenvalues, eigenvectors = eigsh(L, k=3, sigma=0, which='LM')
            idx_sort = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx_sort[1]]
        except Exception:
            fiedler = np.random.randn(m)
            fiedler -= fiedler.mean()
        
        return fiedler
    
    # Main search
    best_count = 0
    best_chosen = None
    best_face_counts = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: MCS-based greedy
    chosen, face_counts, count = mcs_ordering()
    count = greedy_polish(chosen, face_counts)
    
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Strategy 2: Spectral initialization + local search
    fiedler = spectral_init()
    chosen = np.zeros(m, dtype=bool)
    face_counts = np.zeros(num_faces, dtype=np.int32)
    
    edge_order = np.argsort(-fiedler)
    for idx in edge_order:
        idx = int(idx)
        can_add = True
        for f_idx in edge_faces[idx]:
            if face_counts[f_idx] >= 3:
                can_add = False
                break
        if can_add:
            chosen[idx] = True
            for f_idx in edge_faces[idx]:
                face_counts[f_idx] += 1
    
    count = int(chosen.sum())
    count = greedy_polish(chosen, face_counts)
    
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
        best_face_counts = face_counts.copy()
    
    # Local search from best using WalkSAT
    remaining = deadline - time.time() - 10
    if remaining > 5:
        chosen, face_counts, count = walk_sat_local_search(best_chosen, best_face_counts, best_count, 
                                                   max_time=remaining)
        count = greedy_polish(chosen, face_counts)
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
    
    # Multiple restarts with different strategies
    for r in range(8):
        if time.time() > deadline - 15:
            break
        
        random.seed(100 + r * 7)
        np.random.seed(100 + r * 7)
        
        if r % 2 == 0:
            chosen, face_counts, count = mcs_ordering()
        else:
            fiedler = spectral_init()
            fiedler += np.random.randn(m) * 0.1
            chosen = np.zeros(m, dtype=bool)
            face_counts = np.zeros(num_faces, dtype=np.int32)
            edge_order = np.argsort(-fiedler)
            for idx in edge_order:
                idx = int(idx)
                can_add = True
                for f_idx in edge_faces[idx]:
                    if face_counts[f_idx] >= 3:
                        can_add = False
                        break
                if can_add:
                    chosen[idx] = True
                    for f_idx in edge_faces[idx]:
                        face_counts[f_idx] += 1
            count = int(chosen.sum())
        
        count = greedy_polish(chosen, face_counts)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
            best_face_counts = face_counts.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 10:
            chosen, face_counts, count = walk_sat_local_search(chosen, face_counts, count,
                                                       max_time=remaining)
            count = greedy_polish(chosen, face_counts)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
                best_face_counts = face_counts.copy()
    
    # Final polish
    count = greedy_polish(best_chosen, best_face_counts)
    if count > best_count:
        best_count = count
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
