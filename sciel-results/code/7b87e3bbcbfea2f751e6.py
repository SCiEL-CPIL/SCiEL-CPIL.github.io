import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    n = 40
    target_edges = 401
    
    def build_books(A):
        """Compute full book size matrix from adjacency matrix."""
        B = A.astype(np.int32) @ A.astype(np.int32)
        # Mask non-edges to 0
        return np.where(A > 0, B, 0)
    
    def get_max_book(books):
        return int(books.max())
    
    def score_tuple(books):
        max_b = int(books.max())
        count_max = int(np.sum(books == max_b))
        sum_books = int(books.sum())
        return (max_b, count_max, sum_books)
    
    def update_books_on_add(A, books, u, v):
        """Incrementally update books when adding edge (u,v). Returns new books."""
        # books[i,j] = number of common neighbors of i and j
        # Adding edge (u,v) affects books where u or v is involved
        # For any k: books[u,k] increases by A[v,k] (if k != u), books[v,k] increases by A[u,k] (if k != v)
        # Also books[u,v] = common neighbors of u and v (which is what we're computing)
        # Actually, books[u,v] after adding = old_books[u,v] + 0 (since we're adding the edge, 
        # the common neighbors don't change from adding the edge itself)
        # Wait - books[u,v] is the number of triangles containing edge (u,v), which equals
        # the number of common neighbors of u and v. Adding edge (u,v) doesn't change 
        # who the common neighbors are.
        # But adding (u,v) DOES change books for other pairs:
        # For pair (u, k) where k != v: books[u,k] += A[v,k] (because now v is a common neighbor if A[v,k]=1)
        # For pair (v, k) where k != u: books[v,k] += A[u,k]
        # For pair (k1, k2) where neither is u or v: no change
        # Actually wait - books[u,k] = sum_j A[u,j]*A[k,j]. Adding A[u,v]=1 means:
        #   books[u,k] += A[v,k] for all k (including k=v: books[u,v] += A[v,v]=0)
        #   books[v,k] += A[u,k] for all k
        # Let me be more careful. books[i,j] = sum over all m of A[i,m]*A[j,m]
        # When we set A[u,v]=1 (and A[v,u]=1):
        #   For books[u,j] (j != u): sum_m A[u,m]*A[j,m] gains term m=v: A[u,v]*A[j,v] = 1*A[j,v] = A[j,v]
        #   For books[v,j] (j != v): gains A[j,u]
        #   For books[i,j] where i != u,v and j != u,v: no change
        #   For books[u,v]: gains A[u,v]*A[v,v] + ... wait, books[u,v] = sum_m A[u,m]*A[v,m]
        #     Before: sum_m A[u,m]*A[v,m] (with A[u,v]=0)
        #     After: sum_m A[u,m]*A[v,m] + A[u,v]*A[v,v] = same + 0 (since A[v,v]=0)
        #     So books[u,v] doesn't change!
        # Hmm, that's not right either. Let me reconsider.
        # books[u,v] = number of common neighbors of u and v = sum_m A[u,m]*A[v,m] where m != u,v
        # Actually in the matrix product A@A, (A@A)[u,v] = sum_m A[u,m]*A[m,v] = sum_m A[u,m]*A[v,m]
        # When we add A[u,v]=1:
        #   (A@A)[u,v] = sum_m A[u,m]*A[v,m]. The term m=v: A[u,v]*A[v,v] = 1*0 = 0
        #   The term m=u: A[u,u]*A[v,u] = 0*1 = 0
        #   So books[u,v] is unchanged by adding (u,v). Correct.
        #   But for (A@A)[u,j] where j != u,v: the term m=v: A[u,v]*A[v,j] = 1*A[v,j] = A[v,j] is new
        #   So books[u,j] += A[v,j] for j != u, v
        #   Similarly books[v,j] += A[u,j] for j != u, v
        #   And books[j,u] += A[j,v] for j != u,v (symmetric)
        #   And books[j,v] += A[j,u] for j != u,v
        
        new_books = books.copy()
        # books[u,j] += A[v,j] for all j
        new_books[u, :] += A[v, :]
        # books[v,j] += A[u,j] for all j  
        new_books[v, :] += A[u, :]
        # books[j,u] += A[j,v] for all j
        new_books[:, u] += A[:, v]
        # books[j,v] += A[j,v] ... wait, books[j,v] += A[j,u]
        new_books[:, v] += A[:, u]
        # But we double-counted books[u,v]: it got +A[v,v]=0 from row u, +A[u,u]=0 from row v,
        # +A[u,v] from col u... wait let me redo this carefully.
        # Actually the issue is that when we do new_books[u,:] += A[v,:], this includes new_books[u,v] += A[v,v] = 0. Good.
        # When we do new_books[:,u] += A[:,v], this includes new_books[v,u] += A[v,v] = 0. Good.
        # But new_books[u,v] gets contributions from:
        #   new_books[u,:] += A[v,:] -> new_books[u,v] += A[v,v] = 0
        #   new_books[v,:] += A[u,:] -> doesn't affect row u
        #   new_books[:,u] += A[:,v] -> new_books[u,u] += A[u,v]... no wait
        #   new_books[:,u] += A[:,v] means for each row i: new_books[i,u] += A[i,v]
        #     So new_books[u,u] += A[u,v] = 1 (but u,u is diagonal, we don't care)
        #     new_books[v,u] += A[v,v] = 0
        #   new_books[:,v] += A[:,u] means for each row i: new_books[i,v] += A[i,u]
        #     So new_books[u,v] += A[u,u] = 0
        #     new_books[v,v] += A[v,u] = 1 (diagonal, don't care)
        # So new_books[u,v] = old_books[u,v] + 0 + 0 + 0 + 0 = old_books[u,v]. Correct!
        # But wait, I think I'm overcomplicating. Let me just recompute for the affected rows/cols.
        # Actually, the cleanest way: only recompute books for rows/cols involving u or v.
        # books[u,j] for all j: recompute as sum_m A[u,m]*A[j,m]
        # books[v,j] for all j: recompute
        # books[i,u] for all i: recompute (same as books[u,i] by symmetry)
        # books[i,v] for all i: recompute
        # Since the matrix is symmetric, we just need to update rows u and v.
        
        # Recompute row u: books[u,j] = sum_m A[u,m] * A[j,m] = A[u,:] @ A[:,j] for each j
        # This is just A[u,:] @ A (row vector times matrix)
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        # By symmetry, columns are updated automatically if we maintain symmetry
        # But to be safe, also set the columns:
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        
        # Mask non-edges to 0
        new_books = np.where(A > 0, new_books, 0)
        return new_books
    
    def update_books_on_remove(A, books, u, v):
        """Incrementally update books when removing edge (u,v)."""
        # Similar logic: recompute rows u and v after setting A[u,v]=A[v,u]=0
        new_books = books.copy()
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        new_books = np.where(A > 0, new_books, 0)
        return new_books
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    best_score = None
    best_A = None
    best_books = None
    
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random Init
        random.shuffle(all_pairs)
        edges = all_pairs[:target_edges]
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        books = build_books(A)
        cur_score = score_tuple(books)
        cur_max = cur_score[0]
        
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
            best_books = books.copy()
            
        if cur_max <= 7:
            break
            
        # Local Search with incremental updates
        stuck = 0
        max_iterations = 2000
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find edges with maximum book size
            max_val = int(books.max())
            if max_val <= 7:
                break
            
            # Get all max-book edges
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and books[i, j] == max_val:
                        max_edges.append((i, j))
                        
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            # Strategy: try to remove (u,v) and add a non-edge (x,y)
            # Prefer non-edges where x or y is a common neighbor of u and v,
            # or where the swap reduces the book of (u,v)'s neighbors
            
            # Get common neighbors of u and v
            common_neighbors = []
            for k in range(n):
                if k != u and k != v and A[u, k] == 1 and A[v, k] == 1:
                    common_neighbors.append(k)
            
            # Build candidate non-edges
            # Priority 1: edges between common neighbors (to break triangles)
            # Priority 2: edges from common neighbors to non-neighbors of u or v
            # Priority 3: random non-edges
            
            candidates = []
            # Add edges between common neighbors
            for i_idx in range(len(common_neighbors)):
                for j_idx in range(i_idx+1, len(common_neighbors)):
                    x, y = common_neighbors[i_idx], common_neighbors[j_idx]
                    if A[x, y] == 0:
                        candidates.append((x, y))
            
            # Add edges from common neighbors to random vertices
            for cn in common_neighbors:
                for _ in range(3):
                    other = random.randint(0, n-1)
                    if other != cn and A[cn, other] == 0:
                        x, y = min(cn, other), max(cn, other)
                        if x != u or y != v:  # Don't re-add the same edge
                            candidates.append((x, y))
            
            # Add some random non-edges
            for _ in range(30):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and A[x, y] == 0:
                    x, y = min(x, y), max(x, y)
                    if x != u or y != v:
                        candidates.append((x, y))
            
            random.shuffle(candidates)
            
            improved = False
            for x, y in candidates[:50]:
                if A[x, y] == 1:
                    continue
                    
                # Perform swap: remove (u,v), add (x,y)
                A[u, v] = 0
                A[v, u] = 0
                books = update_books_on_remove(A, books, u, v)
                
                A[x, y] = 1
                A[y, x] = 1
                books = update_books_on_add(A, books, x, y)
                
                new_score = score_tuple(books)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_max = new_score[0]
                    improved = True
                    if best_score is None or cur_score < best_score:
                        best_score = cur_score
                        best_A = A.copy()
                        best_books = books.copy()
                    break
                else:
                    # Revert
                    A[x, y] = 0
                    A[y, x] = 0
                    books = update_books_on_remove(A, books, x, y)
                    
                    A[u, v] = 1
                    A[v, u] = 1
                    books = update_books_on_add(A, books, u, v)
            
            if improved:
                stuck = 0
            else:
                stuck += 1
                if stuck > 30:
                    break
    
    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        existing = set(tuple(e) for e in edges)
        if len(edges) > target_edges:
            while len(edges) > target_edges:
                idx = random.randint(0, len(edges)-1)
                e = edges.pop(idx)
                existing.discard(tuple(e))
                existing.discard(tuple(reversed(e)))
        else:
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
                    
    return edges
