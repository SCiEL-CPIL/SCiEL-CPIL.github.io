        best = None
        # Try MRV first
        col = try_color_mrv(n, adj, lists)
        if valid(col, lists, edges):
            best = col
        else:
            if best is None: best = col
            
        if best is None or not valid(best, lists, edges):
             # Try static orders
             for order in orders:
                 col = try_color(n, adj, lists, order)
                 if valid(col, lists, edges):
                     best = col
                     break
                 if best is None:
                     best = col
                 if deadline():
                     break
