            # Reduce
            current_edges = [list(e) for e in edges]
            while True:
                removed_any = False
                # We need to be careful with indices.
                # If we remove index i, the next edge is at i.
                # So we can just continue to i (which is now the next element).
                # But we need to re-check? No, if edge i was redundant, removing it
                # doesn't make previous edges redundant (they were already checked and kept).
                # It might make *later* edges redundant? Yes.
                # So we can just continue the loop.
                i = 0
                while i < len(current_edges):
                    if time.time() - start_time > 105:
                        break
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable(n, test_edges):
                        current_edges = test_edges
                        removed_any = True
                        # Do not increment i, check the next edge (which is now at i)
                    else:
                        i += 1
                if not removed_any:
                    break
