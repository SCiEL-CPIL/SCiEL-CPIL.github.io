<<<<<<< SEARCH
        max_no_progress = 30
        no_progress = 0
        
        while time_ok():
            # Phase 1: Try to add edges
            if try_add_any_edge():
                no_progress = 0
                continue
            
            # Phase 2: Try to recolor
            if try_recolor_any_edge():
                no_progress += 1
                if no_progress >= max_no_progress:
                    break
                continue
            
            # Phase 3: Try edge swap
            if try_edge_swap():
                no_progress = 0
                continue
            
            # Stuck
            break
        
        return cur_edges
=======
        max_no_progress = 50
        no_progress = 0
        
        while time_ok():
            # Phase 1: Try to add edges
            if try_add_any_edge():
                no_progress = 0
                continue
            
            # Phase 2: Try to recolor
            if try_recolor_any_edge():
                no_progress += 1
                if no_progress >= max_no_progress:
                    break
                continue
            
            # Phase 3: Try edge swap
            if try_edge_swap():
                no_progress = 0
                continue
            
            # Stuck
            break
        
        return cur_edges
>>>>>>> REPLACE

