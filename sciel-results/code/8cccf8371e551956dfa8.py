<<<<<<< SEARCH
                # Efficient girth check
                if _check_girth_fast(m, a, b, g):
=======
                # Efficient girth check
                result = _check_girth_fast(m, a, b, g, deadline - 2.0)
                if result is None:
                    break  # time limit, stop searching
                if result:
>>>>>>> REPLACE

And I need to remove the old `_has_nonbacktracking_closed_walk` function since it's no longer used. Let me check if it's still referenced... It was only called from `_check_girth_fast` which I've replaced. Let me remove it.

