import inspect, importlib, sys, os, json

# Try to import the module the baseline uses
try:
    mod = importlib.import_module("autoresearch_suite.science_tcs_expansion")
    print("MODULE:", mod.__file__)
    print("DIR:", [n for n in dir(mod) if not n.startswith('__')])
    src = inspect.getsource(mod.pebble_baseline)
    print("=== pebble_baseline source ===")
    print(src)
except Exception as e:
    print("IMPORT ERROR:", type(e).__name__, e)
