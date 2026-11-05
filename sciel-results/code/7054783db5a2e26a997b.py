"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx
    
    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    n_k4s = len(k4_edges)

    def is_violated(k4_idx, colours):
        edges = k4_edges[k4_idx]
        return len(set(colours[e] for e in edges)) < 5

    def get_violations(colours):
        violated = set()
        for k4_idx in range(n_k4s):
            if is_violated(k4_idx, colours):
                violated.add(k4_idx)
        return violated

    def local_search(colours, max_colors, deadline):
        violated = get_violations(colours)
        steps = 0
        max_steps = 50000  # Hard cap to prevent infinite loops in bad trials
        
        while violated and steps < max_steps and time.time() < deadline:
            steps += 1
            # Pick a random violated K4
            k4_idx = random.choice(list(violated))
            edges = k4_edges[k4_idx]
            
            # Current colors in this K4
            current_colors = set(colours[e] for e in edges)
            
            # We need to change one edge to a color NOT in current_colors
            # to ensure this K4 becomes valid (it will have at least len(current_colors)+1 colors, 
            # wait. If we change an edge to a new color, the set of colors becomes 
            # (current_colors - {old_color_of_e}) U {new_color}.
            # If new_color is not in current_colors, the size increases by 1 if old_color was unique in the set?
            # No. If old_color appears multiple times, removing one instance doesn't remove the color from the set.
            # So size becomes max(|current_colors|, |current_colors - {old}| + 1).
            # Since |current_colors| < 5, we want to reach 5.
            # If |current_colors| = 4, we need to add a 5th color.
            # If we pick new_color not in current_colors, the new set will have size 5 
            # UNLESS old_color was the only one of its kind AND we remove it?
            # No, we are REPLACING the color of edge e.
            # Old set S. e had color c_old.
            # New set S' = (S \ {c_old}) U {c_new}.
            # If c_new not in S, and c_old in S:
            #   If c_old appears only once in the 6 edges, S loses c_old. Size decreases by 1, then increases by 1. Net 0?
            #   Wait. S is the set of colors.
            #   If c_old is unique in the edges of this K4, then S' = S - {c_old} + {c_new}.
            #   If c_new not in S, then |S'| = |S|. This does NOT fix the violation if |S|=4.
            #   We need |S'| >= 5.
            #   So we must ensure that the new color adds a distinct color.
            #   If we pick c_new not in S, then c_new is definitely in S'.
            #   Is c_old still in S'? Only if some other edge has c_old.
            #   If some other edge has c_old, then c_old is in S'. So S' = S U {c_new}. Size increases by 1.
            #   If NO other edge has c_old, then c_old is removed from S'. S' = (S \ {c_old}) U {c_new}.
            #   Since c_new not in S, c_new not in S\{c_old}.
            #   So |S'| = |S|.
            #   So, to guarantee an increase, we should prefer changing an edge whose color is shared by another edge in the K4.
            #   Or, simply, if |S| < 5, we can just pick any edge and change it to a color not in S.
            #   If the size doesn't increase, we might still fix it if |S| was 4 and we accidentally increased? No.
            #   Let's check:
            #   Case 1: |S| = 4.
            #   If we change edge e (color c) to c_new (not in S).
            #   If c is unique in K4: New set size is 4. Still violated.
            #   If c is not unique in K4: New set size is 5. Fixed.
            #   So, we should prioritize edges whose color is NOT unique in the K4.
            
            # Find edges in this K4 that can be changed to increase color count
            color_counts = {}
            for e in edges:
                c = colours[e]
                color_counts[c] = color_counts.get(c, 0) + 1
            
            candidate_edges = []
            for e in edges:
                c = colours[e]
                if color_counts[c] > 1:
                    candidate_edges.append(e)
            
            # If no candidate edges (all colors unique, but that means |S|=6, which is not violated),
            # this branch is only entered if violated, so |S| < 5.
            # If |S| < 5, by Pigeonhole, some color must appear at least twice?
            # 6 edges, < 5 colors. Yes, at least one color appears >= 2 times.
            # So candidate_edges is never empty.
            
            e = random.choice(candidate_edges)
            
            # Pick a new color not in current_colors
            available_colors = [c for c in range(max_colors) if c not in current_colors]
            if not available_colors:
                # This means max_colors <= len(current_colors).
                # But we are trying max_colors >= 13, and len(current_colors) <= 4.
                # So this should never happen.
                continue
                
            new_color = random.choice(available_colors)
            
            # Apply change
            old_color = colours[e]
            colours[e] = new_color
            
            # Update violations for all K4s containing edge e
            for k4 in edge_to_k4s[e]:
                was_violated = k4 in violated
                is_now_violated = is_violated(k4, colours)
                
                if was_violated and not is_now_violated:
                    violated.discard(k4)
                elif not was_violated and is_now_violated:
                    violated.add(k4)
                # If was_violated and is_now_violated, no change
                # If not was_violated and not is_now_violated, no change

        return len(violated) == 0, colours

    # Try different max_colors starting from the lower bound
    for max_colors in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        # We can increase trials if local search is fast
        for trial in range(10):
            if time.time() > deadline:
                break
                
            # Random assignment using colors 0..max_colors-1
            # To help the search, we might want to ensure all colors are used initially?
            # Or just random.
            colours = [random.randint(0, max_colors - 1) for _ in range(n_edges)]
            
            success, colours = local_search(colours, max_colors, deadline)
            
            if success:
                # Verify once more just in case
                if not get_violations(colours):
                    return colours
    
    # Fallback: 17-color solution
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
