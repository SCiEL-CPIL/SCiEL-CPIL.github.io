    for rb in range(2):
        for cb in range(3):
            add_strassen_term(rb*2, 0, cb*2)
            add_strassen_term(rb*2, 2, cb*2)
