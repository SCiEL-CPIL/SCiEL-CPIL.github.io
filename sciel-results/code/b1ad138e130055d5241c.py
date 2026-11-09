import time
import random
import numpy as np

_DEADLINE = 150.0  # evaluator wall-clock limit
_SAFETY = 15.0     # margin so we return well before the kill


def _build_edges(d):
    """Return list of (u, v, dim) for all edges of Q_d, u < v."""
    edges = []
    n = 1 << d
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y, i))
    return edges


def _build_faces(d):
    """
    For each 2D face (identified by a vertex x with bits a,b = 0 and dims a<b),
    return the 4 edge indices forming that face.
    Also build edge->faces adjacency.
    """
    n = 1 << d
    # Map (u,v) to edge index
    edge_idx = {}
    all_edges = _build_edges(d)
    for idx, (u, v, dim) in enumerate(all_edges):
        edge_idx[(u, v)] = idx

    faces = []  # list of list of 4 edge indices
    edge_to_faces = [[] for _ in all_edges]

    for x in range(n):
        for a in range(d):
            if (x >> a) & 1:
                continue
            for b in range(a + 1, d):
                if (x >> b) & 1:
                    continue
                # Face vertices: x, x^ea, x^eb, x^ea^eb
                x1 = x ^ (1 << a)
                x2 = x ^ (1 << b)
                x3 = x ^ (1 << a) ^ (1 << b)
                # 4 edges: (x,x1) dim a, (x,x2) dim b, (x1,x3) dim b, (x2,x3) dim a
                e0 = edge_idx[(x, x1)]
                e1 = edge_idx[(x, x2)]
                e2 = edge_idx[(x1, x3)]
                e3 = edge_idx[(x2, x3)]
                faces.append([e0, e1, e2, e3])
                for e in [e0, e1, e2, e3]:
                    edge_to_faces[e].append(len(faces) - 1)

    return all_edges, faces, edge_to_faces


def _is_valid(selected, faces):
    """Check if no face has all 4 edges selected."""
    for f in faces:
        if all(selected[e] for e in f):
            return False
    return True


def _count_edges(selected):
    """Count number of selected edges."""
    return sum(selected)


def _random_valid_start(all_edges, faces, edge_to_faces):
    """
    Start with all edges, then remove edges to break all C4s.
    This is a greedy set cover / hitting set approach.
    """
    selected = [True] * len(all_edges)
    
    # While there is a face with all 4 edges selected, remove one
    # To make it more effective, we can pick the edge that is in the most "conflicting" faces?
    # Or just random. Random is faster.
    # We maintain a list of "bad" faces.
    
    # Initial bad faces
    bad_faces = []
    for i, f in enumerate(faces):
        if all(selected[e] for e in f):
            bad_faces.append(i)
            
    while bad_faces:
        if (time.time() - _t0_global) > _DEADLINE - _SAFETY:
            break
        # Pick a random bad face
        fi = random.choice(bad_faces)
        f = faces[fi]
        
        # Pick a random edge in this face to remove
        # To improve, we could pick the edge that appears in the most other bad faces
        # For now, random removal
        e_to_remove = random.choice(f)
        selected[e_to_remove] = False
        
        # Update bad_faces list
        # This is O(N_faces) per removal which might be slow.
        # Better: just rebuild the list of bad faces occasionally or use a counter.
        # Given 448 edges and ~672 faces, rebuilding is fast (~1ms).
        # Let's rebuild every 10 removals or just check validity at the end of the loop?
        # No, we need to ensure we break *all* faces.
        
        # Efficient update:
        # The face fi is no longer bad.
        # Other faces containing e_to_remove might become valid (if they were bad, they are now not bad? No, they have one less edge, so they are definitely not bad).
        # Wait, if a face was bad (4 edges), removing one makes it have 3 edges, so it's NOT bad.
        # So we only need to remove faces from bad_faces that contain e_to_remove.
        # But we don't have a face->edges index easily accessible for removal from a list.
        
        # Let's just rebuild the bad_faces list. It's 672 checks. Very fast.
        bad_faces = [i for i, f in enumerate(faces) if all(selected[e] for e in f)]

    return selected


def _local_search(selected, faces, edge_to_faces, all_edges):
    """
    Local search: try to add edges. If an edge e cannot be added because of conflict in face f,
    try to remove one of the other edges in f to make room for e.
    This is a 1-swap (remove 1, add 1).
    """
    num_edges = len(all_edges)
    improved = True
    while improved and (time.time() - _t0_global) < _DEADLINE - _SAFETY:
        improved = False
        # Shuffle edges to avoid bias
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if (time.time() - _t0_global) > _DEADLINE - _SAFETY:
                return selected
            
            if selected[e]:
                continue
                
            # Check if e can be added
            conflicts = []
            for fi in edge_to_faces[e]:
                f = faces[fi]
                # Check if other 3 are selected
                others = [x for x in f if x != e]
                if all(selected[x] for x in others):
                    conflicts.append(fi)
                    
            if not conflicts:
                # Can add freely
                selected[e] = True
                improved = True
            else:
                # Try to swap: remove one edge from a conflicting face to allow e
                # Pick a random conflict
                fi = random.choice(conflicts)
                f = faces[fi]
                others = [x for x in f if x != e]
                
                # Try removing each edge in others
                random.shuffle(others)
                for x in others:
                    # If we remove x, does it break any other faces?
                    # It might break other faces that were relying on x, but that's fine (we just want to add e).
                    # However, removing x might make some other edge addable? No, we only care about count.
                    # We need to ensure that after removing x, e can be added.
                    # Since x is in the face f that was blocking e, removing x unblocks f.
                    # Are there other faces blocking e?
                    # If there are other conflicting faces for e, removing x from f doesn't help if another face g still has all 3 others selected.
                    # So we should only try this if this was the *only* conflict?
                    # Or we can just try the swap and check validity?
                    # Actually, if e has multiple conflicts, removing one edge from one face doesn't guarantee e can be added.
                    # Let's check if e has exactly 1 conflict.
                    if len(conflicts) == 1:
                        # Remove x, add e
                        selected[x] = False
                        selected[e] = True
                        improved = True
                        break
                    else:
                        # Multiple conflicts. A single swap might not work if another face is still fully selected.
                        # We could try a 2-swap, but that's more complex.
                        # For now, skip or try random removal and see if it helps?
                        # Let's try removing x anyway, maybe it reduces the "pressure" for future steps?
                        # But we must maintain validity.
                        # If we remove x, the face f is broken. But if another face g is still fully selected, the state is invalid.
                        # So we can only perform the swap if it results in a valid state.
                        # Since we only removed x, the only faces affected are those containing x.
                        # Faces not containing x were valid before and remain valid.
                        # Faces containing x:
                        # 1. Face f: was fully selected, now has 3 (e is not in it yet? No, e is in f).
                        #    Wait, e is in f. If we remove x (which is in f), f has 3 selected edges (the other 2 + e? No, e is not selected yet).
                        #    Current state: e is NOT selected. f has 4 selected edges (x and 3 others).
                        #    We want to select e.
                        #    If we remove x and select e:
                        #    f will have: (3 others - x) + e = 3 edges. Valid.
                        #    What about other faces containing x?
                        #    They had 4 edges? No, they were valid, so they had <= 3 edges.
                        #    Removing x reduces their count, so they remain valid.
                        #    What about other faces containing e?
                        #    They were valid (<=3 edges). Adding e increases count.
                        #    If a face g containing e had 3 other edges selected, adding e makes it 4 (Invalid).
                        #    So we must ensure that NO other face containing e is fully selected (except f which we fixed).
                        #    We already checked conflicts. If len(conflicts) == 1, then no other face is fully selected.
                        #    So the swap is safe.
                        pass
                        
    return selected


_t0_global = 0.0

def construct():
    global _t0_global
    _t0_global = time.time()
    d = 7
    n = 1 << d

    all_edges, faces, edge_to_faces = _build_faces(d)
    num_edges = len(all_edges)
    
    best_selected = None
    best_count = 0
    
    # Multi-start local search
    # We do several iterations of: Random Valid Start -> Local Search
    # Each start is cheap (greedy removal). Local search is the main cost.
    
    while (time.time() - _t0_global) < _DEADLINE - _SAFETY:
        # 1. Generate a random valid starting set
        selected = _random_valid_start(all_edges, faces, edge_to_faces)
        
        # 2. Local search to improve
        selected = _local_search(selected, faces, edge_to_faces, all_edges)
        
        count = _count_edges(selected)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            
        # Optional: Try to add more edges with a simple greedy pass
        # The local search above already does this, but a final greedy pass is cheap.
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        for e in edge_order:
            if (time.time() - _t0_global) > _DEADLINE - _SAFETY:
                break
            if selected[e]:
                continue
            can_add = True
            for fi in edge_to_faces[e]:
                f = faces[fi]
                if all(selected[x] for x in f if x != e):
                    can_add = False
                    break
            if can_add:
                selected[e] = True
        
        count = _count_edges(selected)
        if count > best_count:
            best_count = count
            best_selected = selected[:]

    # Final validation
    if best_selected is None:
        # Fallback to parity construction
        best_selected = [False] * num_edges
        for idx, (u, v, dim) in enumerate(all_edges):
            src = u if ((u >> dim) & 1) == 0 else v
            if (src.bit_count()) % 2 == 0:
                best_selected[idx] = True
        best_count = _count_edges(best_selected)
    elif not _is_valid(best_selected, faces):
        # Should not happen, but fallback just in case
        best_selected = [False] * num_edges
        for idx, (u, v, dim) in enumerate(all_edges):
            src = u if ((u >> dim) & 1) == 0 else v
            if (src.bit_count()) % 2 == 0:
                best_selected[idx] = True
        best_count = _count_edges(best_selected)

    # Build output
    out = []
    for idx, (u, v, dim) in enumerate(all_edges):
        if best_selected[idx]:
            out.append([u, v])

    return out
