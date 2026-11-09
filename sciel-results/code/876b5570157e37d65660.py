for d in range(3, 15):
    if d % 2 == 1:
        girth_min = d + 5
    else:
        girth_min = d + 6
    if girth_min < g:
        continue
    # ...
