    strassen_2x2(2, 3, 2, 3, 2, 3)
    for i in [2, 3]:
        for k in [2, 3]:
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    for i in [2, 3]:
        for j in [2, 3]:
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    for i in [2, 3]:
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    for j in [2, 3]:
        for k in [2, 3]:
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    for k in [2, 3]:
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    for j in [2, 3]:
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})
    